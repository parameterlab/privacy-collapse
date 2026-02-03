import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import joblib
from vllm import LLM
from vllm.lora.request import LoRARequest

# Import EasySteer components
from easysteer.steer import extract_pca_control_vector, StatisticalControlVector

# ==========================================
# 1. Configuration & Paths
# ==========================================
BASE_MODEL_PATH = "models--llama-3.1/hf/8B-Instruct/"
LORA_ROOT_DIR = os.path.abspath("ft_adapters")
VECTOR_SAVE_DIR = "vectors"
FIGURE_SAVE_DIR = "figures"
MAX_LORA_RANK = 64  
SUBSET_SIZE = 20

os.makedirs(VECTOR_SAVE_DIR, exist_ok=True)
os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)

# ==========================================
# 2. Dataset Setup
# ==========================================

# --- Dataset A: PrivacyLens (Existing) ---
try:
    privacylens_prompts = joblib.load("./privacylens.joblib")
    privacylens_prompts = sorted(privacylens_prompts, key=len)[:SUBSET_SIZE]
except FileNotFoundError:
    print("Warning: privacylens.joblib not found. Using dummy data for Set A.")
    privacylens_prompts = ["Dummy prompt A" for _ in range(5)]

DATASET_A_POS = [f"{i}\nAnswer: B" for i in privacylens_prompts]
DATASET_A_NEG = [f"{i}\nAnswer: A" for i in privacylens_prompts]
ALL_TEXTS_A = DATASET_A_POS + DATASET_A_NEG
POS_INDICES_A = list(range(len(DATASET_A_POS)))
NEG_INDICES_A = list(range(len(DATASET_A_POS), len(ALL_TEXTS_A)))

# --- Dataset B: Manual List (New) ---
MANUAL_PROMPTS = [
"Sammy wanted to go to where the people were. Where might he go?\nOptions:\n(A)the desert\n(B)populated areas",
"The skin was cut while chopping onions, where was this skin likely cut?\nOptions:\n(A)meat\n(B)finger",
"At the end of your meal what will a waiter do?\nOptions:\n(A)set table\n(B)present bill",
"I'm looking for alcohol and fried foods, any suggestions?\nOptions:\n(A)store\n(B)pub",
"Sam went to Paris where he ordered a blowfish at a sushi place. Where was Sam?\nOptions:\n(A)Cuba\n(B)Paris"
"East coast areas such as Florida are found where?\nOptions:\n(A)denmark\n(B)united states",
"If you want to make a big splash in a river with a rock, from where should you throw it?\nOptions:\n(A)ground\n(B)bridge",
"What is a person who is well educated?\nOptions:\n(A)illiterate\n(B)knowledgable",
"What is a good result of jogging?\nOptions:\n(A)heart attacks\n(B)improved health",
"The game was on sale. I got it then because it was usually what?\nOptions:\n(A)overpriced\n(B)expensive",
"What do people perform if they have a band?\nOptions:\n(A)tricks\n(B)music",
"In what geological feature will you find fungus growing?\nOptions:\n(A)basement\n(B)cave",
"Where do you dip a spoon into?\nOptions:\n(A)glass of iced tea\n(B)bowl of soup",
"Where do you keep a book to read before bed?\nOptions:\n(A)coffee table\n(B)bedside table",
"An American might hike a mountain north to get to what?\nOptions:\n(A)switzerland\n(B)canada",
"Sometimes when people think some wrongdoing has ocurred in government, how do they react?\nOptions:\n(A)cook food\n(B)demand justice",
"Where is beaver likely to be found?\nOptions:\n(A)zoo\n(B)lake or river",
"What does a doctor have to do before he is allowed to take an exam to get his MD?\nOptions:\n(A)resuscitate\n(B)study medicine",
"If I find liquid water, what planet am I likely on?\nOptions:\n(A)moon\n(B)planet earth",
"Even drinking was difficult after the surgery, he was having trouble doing what?\nOptions:\n(A)coloring\n(B)swallowing",
]

# Constructing contrast pairs (Answer B vs Answer A) for the manual list
DATASET_B_POS = [f"{i}\nAnswer: B" for i in MANUAL_PROMPTS]
DATASET_B_NEG = [f"{i}\nAnswer: A" for i in MANUAL_PROMPTS]
ALL_TEXTS_B = DATASET_B_POS + DATASET_B_NEG
POS_INDICES_B = list(range(len(DATASET_B_POS)))
NEG_INDICES_B = list(range(len(DATASET_B_POS), len(ALL_TEXTS_B)))

# ==========================================
# 3. Custom Utils
# ==========================================

def get_hiddens_with_lora(llm, texts, lora_req=None):
    """
    Extracts hidden states safely by processing texts one-by-one.
    """
    all_samples_hiddens = []
    
    for text in texts:
        llm.llm_engine.engine_core.collective_rpc("enable_hidden_states_capture")
        outputs = llm.embed([text], lora_request=lora_req)
        
        from vllm.hidden_states import deserialize_hidden_states
        results = llm.llm_engine.engine_core.collective_rpc("get_captured_hidden_states")
        
        llm.llm_engine.engine_core.collective_rpc("clear_hidden_states")
        llm.llm_engine.engine_core.collective_rpc("disable_hidden_states_capture")

        if not results or len(results) == 0:
            continue

        hidden_states_dict = deserialize_hidden_states(results[0])
        sorted_layer_ids = sorted(hidden_states_dict.keys())
        
        # Format: [layer_idx] -> Tensor(num_tokens, hidden_dim)
        sample_layers = [hidden_states_dict[layer_id] for layer_id in sorted_layer_ids]
        sample_layers_cpu = [t.cpu() for t in sample_layers]
        
        all_samples_hiddens.append(sample_layers_cpu)
        
    return all_samples_hiddens

def compute_and_save_vector(hiddens, pos_indices, neg_indices, save_name):
    """Helper to extract and save the control vector."""
    control_vector = extract_pca_control_vector(
        all_hidden_states=hiddens,
        positive_indices=pos_indices,
        negative_indices=neg_indices,
        model_type="llama3",
        token_pos=-1
    )
    save_path = os.path.join(VECTOR_SAVE_DIR, f"{save_name}.gguf")
    control_vector.export_gguf(save_path)
    return save_path

# ==========================================
# 4. Main Execution
# ==========================================

def main():
    print(">>> Initializing vLLM Engine...")
    llm = LLM(
        model=BASE_MODEL_PATH,
        task="embed", 
        enable_lora=True,
        max_lora_rank=MAX_LORA_RANK,
        max_model_len=4096, 
        enforce_eager=True,
        tensor_parallel_size=1,
        enable_chunked_prefill=False 
    )

    # Gather checkpoints
    checkpoints = [d for d in os.listdir(LORA_ROOT_DIR) if os.path.isdir(os.path.join(LORA_ROOT_DIR, d))]
    def sort_key(x):
        if "step" in x: return int(x.replace("step", ""))
        if "final" in x: return 999999
        return 0
    checkpoints.sort(key=sort_key)
    
    # Queue: (Name, Path)
    processing_queue = [("base", None)] + [(ckpt, os.path.join(LORA_ROOT_DIR, ckpt)) for ckpt in checkpoints]
    
    # Store paths to generated vectors for analysis
    # Structure: {"base": {"A": path, "B": path}, "ckpt1": ...}
    generated_vectors = {}

    for name, path in processing_queue:
        print(f"\n>>> Processing Model: {name}")
        generated_vectors[name] = {}
        
        req = None
        if path:
            req = LoRARequest(name, abs(hash(name)) % 100000, path)
            
        # --- Process Dataset A (PrivacyLens) ---
        print(f"   Extracting hiddens for Dataset A...")
        hiddens_a = get_hiddens_with_lora(llm, ALL_TEXTS_A, lora_req=req)
        path_a = compute_and_save_vector(hiddens_a, POS_INDICES_A, NEG_INDICES_A, f"{name}_privacy")
        generated_vectors[name]["A"] = path_a
        print(f"   Vector A saved: {path_a}")
        
        del hiddens_a
        gc.collect()

        # --- Process Dataset B (Manual) ---
        print(f"   Extracting hiddens for Dataset B...")
        hiddens_b = get_hiddens_with_lora(llm, ALL_TEXTS_B, lora_req=req)
        path_b = compute_and_save_vector(hiddens_b, POS_INDICES_B, NEG_INDICES_B, f"{name}_gsm8k")
        generated_vectors[name]["B"] = path_b
        print(f"   Vector B saved: {path_b}")

        del hiddens_b
        gc.collect()

    # Clean up LLM to free memory for analysis
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # ==========================================
    # 5. Analysis & Plotting
    # ==========================================
    perform_analysis(generated_vectors)

def perform_analysis(vector_map):
    print("\n>>> Analyzing Vector Evolution...")
    
    if "base" not in vector_map:
        print("Error: Base vector missing, cannot compare.")
        return

    base_vec_a = StatisticalControlVector.import_gguf(vector_map["base"]["A"])
    base_vec_b = StatisticalControlVector.import_gguf(vector_map["base"]["B"])
    
    layers = sorted(base_vec_a.directions.keys())
    
    # Initialize Plot Data Containers
    # format: { "model_name": [val_layer_0, val_layer_1, ...] }
    sim_data_a = {}
    norm_data_a = {}
    sim_data_b = {}
    norm_data_b = {}

    # Iterate over all models to compute metrics
    for model_name, paths in vector_map.items():
        # Load vectors
        vec_a = StatisticalControlVector.import_gguf(paths["A"])
        vec_b = StatisticalControlVector.import_gguf(paths["B"])
        
        sims_a, norms_a = [], []
        sims_b, norms_b = [], []

        for layer in layers:
            if layer in vec_a.directions and layer in base_vec_a.directions:
                # Set A Comparisons
                v_base_a = base_vec_a.directions[layer].flatten()
                v_curr_a = vec_a.directions[layer].flatten()
                
                # Cosine Similarity vs Base
                sim_a = 1 - cosine(v_base_a, v_curr_a)
                sims_a.append(sim_a)
                
                # Norm (Magnitude)
                norms_a.append(np.linalg.norm(v_curr_a))

                # Set B Comparisons
                v_base_b = base_vec_b.directions[layer].flatten()
                v_curr_b = vec_b.directions[layer].flatten()
                
                sim_b = 1 - cosine(v_base_b, v_curr_b)
                sims_b.append(sim_b)
                norms_b.append(np.linalg.norm(v_curr_b))
        
        sim_data_a[model_name] = sims_a
        norm_data_a[model_name] = norms_a
        sim_data_b[model_name] = sims_b
        norm_data_b[model_name] = norms_b

    # --- Plotting Helper ---
    def plot_metric(data_dict, title, ylabel, filename):
        plt.figure(figsize=(10, 6))
        for model_name, values in data_dict.items():
            # Highlight base with a thick black line, others colored
            style = '--' if model_name == 'base' else '-'
            alpha = 1.0 if model_name == 'base' or 'final' in model_name else 0.6
            width = 2.5 if model_name == 'base' else 1.5
            
            plt.plot(layers, values, label=model_name, linestyle=style, alpha=alpha, linewidth=width)
            
        plt.title(title)
        plt.xlabel("Layer Index")
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_SAVE_DIR, filename))
        plt.close()

    # Generate the 4 requested plots
    print("   Generating plots...")
    
    # 1. Similarity Plots
    plot_metric(sim_data_a, "Cosine Similarity to Base (Dataset A)", "Cosine Similarity", "similarity_set_A.png")
    plot_metric(sim_data_b, "Cosine Similarity to Base (Dataset B)", "Cosine Similarity", "similarity_set_B.png")
    
    # 2. Norm Plots
    plot_metric(norm_data_a, "Vector Norm Evolution (Dataset A)", "L2 Norm", "norm_set_A.png")
    plot_metric(norm_data_b, "Vector Norm Evolution (Dataset B)", "L2 Norm", "norm_set_B.png")
    
    print(f"Analysis complete. Figures saved to {FIGURE_SAVE_DIR}")

if __name__ == "__main__":
    main()