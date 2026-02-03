import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens, LogitLens
from nethook import TraceDict
import joblib

# --- HELPER FUNCTIONS (Same as before) ---
def get_layer_names(model):
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

def analyze_dataset(model_name, dataset, use_tuned_lens=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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
    
    # Store all raw data here
    all_history = [] 

    print(f"Processing {len(dataset)} examples...")
    
    for item in tqdm(dataset):
        prompt = item['prompt']
        # Note: We add spaces for GPT/Llama tokenizers usually
        token_correct_str = " " + item['correct'] if not item['correct'].startswith(" ") else item['correct']
        token_wrong_str   = " " + item['wrong'] if not item['wrong'].startswith(" ") else item['wrong']

        # Get IDs (add_special_tokens=False is CRITICAL)
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
            del logits, hidden

    # Convert to DataFrame for easy aggregation
    df = pd.DataFrame(all_history)
    
    # --- AGGREGATION FOR PAPER ---
    summary = df.groupby("layer").agg({
        "rank_correct": ["mean", "median"],
        "rank_wrong": ["mean", "median"],
        "prob_correct": "mean",
        "prob_wrong": "mean"
    })
    
    # Flatten columns for clean printing
    summary.columns = ['_'.join(col) for col in summary.columns]
    
    print("\n" + "="*80)
    print(f"AGGREGATED RESULTS ({len(dataset)} Prompts) - {model_name}")
    print("="*80)
    print(f"{'Layer':<6} | {'Med Rank (Cor)':<15} {'Mean Prob (Cor)':<16} | {'Med Rank (Wro)':<15} {'Mean Prob (Wro)':<16}")
    print("-" * 80)
    
    for layer in summary.index:
        r_c = summary.loc[layer, 'rank_correct_median']
        p_c = summary.loc[layer, 'prob_correct_mean']
        r_w = summary.loc[layer, 'rank_wrong_median']
        p_w = summary.loc[layer, 'prob_wrong_mean']
        
        print(f"{layer:<6} | {r_c:<15.1f} {p_c:<16.4f} | {r_w:<15.1f} {p_w:<16.4f}")
        
    return summary

if __name__ == "__main__":
    # Example Dataset (MMLU Style or Fact Retrieval)
    # You can load this from a JSON file in your actual workflow
    privacylens = joblib.load("../privacylens.joblib")
    my_dataset = [{"prompt": f"<|begin_of_text|>{i}[/INST][ASST](", "correct": "B", "wrong": "A"} for i in privacylens]

    df_results = analyze_dataset("models--llama-3.1/hf/8B-Instruct/", my_dataset[:50], use_tuned_lens=False)
    
    # Optional: Save to CSV for plotting in your paper
    df_results.to_csv("base_trajectory_results.csv")