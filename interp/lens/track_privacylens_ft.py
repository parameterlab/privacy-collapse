import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel # <--- Added PEFT
from tuned_lens.nn.lenses import TunedLens, LogitLens
from nethook import TraceDict
import joblib

# --- HELPER FUNCTIONS (Same as before) ---
def get_layer_names(model):
    # This remains the same as long as we merge the adapter
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [f"transformer.h.{i}" for i in range(model.config.n_layer)]
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return [f"gpt_neox.layers.{i}" for i in range(model.config.num_hidden_layers)]
    else:
        raise ValueError("Unknown model architecture")

def get_last_token_hidden(layer_output):
    if isinstance(layer_output, tuple): layer_output = layer_output[0]
    if layer_output.dim() == 2: return layer_output[-1, :]
    return layer_output[0, -1, :]

def calculate_rank(logits, target_token_id):
    probs = torch.softmax(logits, dim=-1)
    target_prob = probs[target_token_id].item()
    rank = torch.sum(probs > target_prob).item() + 1
    return rank, target_prob

# --- MAIN ANALYSIS FUNCTION ---

def analyze_dataset(model_name, adapter_path, dataset, use_tuned_lens=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Base Model: {model_name}...")
    # Use torch_dtype="auto" to ensure half-precision if the model supports it
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map=device
    )
    
    print(f"Loading Adapter: {adapter_path}...")
    # Load the adapter and merge it into the base weights
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload() 
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Note: TunedLens for the base model may not be accurate for an adapter model.
    # LogitLens is usually safer for fine-tuned/adapter models.
    if use_tuned_lens:
        try:
            lens = TunedLens.from_model_and_pretrained(model, lens_resource_id=model_name).to(device)
        except:
            print("Falling back to LogitLens...")
            lens = LogitLens.from_model(model).to(device)
            use_tuned_lens = False
    else:
        lens = LogitLens.from_model(model).to(device)

    trace_layers = get_layer_names(model)
    all_history = [] 

    print(f"Processing {len(dataset)} examples...")
    
    for item in tqdm(dataset):
        prompt = item['prompt']
        token_correct_str = " " + item['correct'] if not item['correct'].startswith(" ") else item['correct']
        token_wrong_str   = " " + item['wrong'] if not item['wrong'].startswith(" ") else item['wrong']

        id_correct = tokenizer.encode(token_correct_str, add_special_tokens=False)[0]
        id_wrong   = tokenizer.encode(token_wrong_str, add_special_tokens=False)[0]
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with TraceDict(model, layers=trace_layers, retain_output=True) as tr:
            model(input_ids)

        for i, layer_name in enumerate(trace_layers):
            hidden = get_last_token_hidden(tr[layer_name].output)
            logits = lens(hidden, idx=i)
            
            rank_corr, prob_corr = calculate_rank(logits, id_correct)
            rank_wrong, prob_wrong = calculate_rank(logits, id_wrong)
            
            all_history.append({
                "layer": i,
                "rank_correct": rank_corr,
                "prob_correct": prob_corr,
                "rank_wrong": rank_wrong,
                "prob_wrong": prob_wrong
            })

    df = pd.DataFrame(all_history)
    summary = df.groupby("layer").agg({
        "rank_correct": ["mean", "median"],
        "rank_wrong": ["mean", "median"],
        "prob_correct": "mean",
        "prob_wrong": "mean"
    })
    summary.columns = ['_'.join(col) for col in summary.columns]
    
    # ... (Print logic remains the same) ...
    return summary

if __name__ == "__main__":
    # Define paths
    BASE_MODEL = "models--llama-3.1/hf/8B-Instruct/"
    ADAPTER_PATH = "../ft_models/final-chkpt" # Local path or HuggingFace ID

    # Example Dataset
    privacylens = joblib.load("../privacylens.joblib")
    my_dataset = [{"prompt": f"<|begin_of_text|>{i}[/INST][ASST](", "correct": "B", "wrong": "A"} for i in privacylens]

    # Run Analysis
    df_results = analyze_dataset(BASE_MODEL, ADAPTER_PATH, my_dataset[:50], use_tuned_lens=False)
    df_results.to_csv("finetuned_trajectory_results.csv")