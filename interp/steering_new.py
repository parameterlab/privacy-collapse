import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import joblib
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import random 
import argparse

# Import EasySteer components
from easysteer.steer import extract_pca_control_vector, StatisticalControlVector

# ==========================================
# 1. Configuration & Paths
# ==========================================
BASE_MODEL_PATH = "models--llama-3.1/hf/8B-Instruct/"
LORA_ROOT_DIR = os.path.abspath("ft_models")
VECTOR_SAVE_DIR = "vectors"
FIGURE_SAVE_DIR = "figures"
MAX_LORA_RANK = 64  
LAYER_TO_PLOT = 29
SUBSET_SIZE = 20

os.makedirs(VECTOR_SAVE_DIR, exist_ok=True)
os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)

# ==========================================
# 2. Dataset Setup
# ==========================================
privacylens_prompts = joblib.load("./privacylens.joblib")
privacylens_prompts = sorted(privacylens_prompts, key=len)[:SUBSET_SIZE]

POSITIVE_DATA = [f"""{i}\nAnswer: B""" for i in privacylens_prompts]
NEGATIVE_DATA = [f"""{i}\nAnswer: A""" for i in privacylens_prompts]

ALL_TEXTS = POSITIVE_DATA + NEGATIVE_DATA
POS_INDICES = list(range(len(POSITIVE_DATA)))
NEG_INDICES = list(range(len(POSITIVE_DATA), len(ALL_TEXTS)))

# ==========================================
# 3. Custom Utils (FIXED)
# ==========================================

def get_hiddens_with_lora(llm, texts, lora_req=None):
    """
    Fixed wrapper to extract hidden states safely.
    Processes texts one-by-one to prevent vLLM reordering/bucketing 
    from mismatching the captured tensors.
    """
    all_samples_hiddens = []
    
    # Iterate one by one to ensure the captured state matches the text
    # This is slightly slower but prevents the crash/mismatch on variable length data
    for text in texts:
        # 1. Enable capture
        llm.llm_engine.engine_core.collective_rpc("enable_hidden_states_capture")
        
        # 2. Run inference on single item
        # We assume texts are strings. Passing a list of 1.
        outputs = llm.embed([text], lora_request=lora_req)
        
        # 3. Retrieve captured tensors
        from vllm.hidden_states import deserialize_hidden_states
        results = llm.llm_engine.engine_core.collective_rpc("get_captured_hidden_states")
        
        # 4. Clean up hooks immediately
        llm.llm_engine.engine_core.collective_rpc("clear_hidden_states")
        llm.llm_engine.engine_core.collective_rpc("disable_hidden_states_capture")

        # 5. Process the result
        if not results or len(results) == 0:
            print(f"Warning: No states captured for text length {len(text)}")
            continue

        hidden_states_dict = deserialize_hidden_states(results[0])
        sorted_layer_ids = sorted(hidden_states_dict.keys())
        
        # Format: [layer_idx] -> Tensor(num_tokens, hidden_dim)
        # Since we ran batch_size=1, the whole tensor belongs to this sample
        sample_layers = [hidden_states_dict[layer_id] for layer_id in sorted_layer_ids]
        
        # Ensure data is on CPU to save VRAM during the loop
        sample_layers_cpu = [t.cpu() for t in sample_layers]
        
        all_samples_hiddens.append(sample_layers_cpu)
        
    return all_samples_hiddens


def plot_pca(hiddens, layer_idx, title, save_path):
    """Generates and saves a PCA plot for a specific model/layer."""
    X = []
    labels = []
    
    # Safety check
    if not hiddens:
        print("Skipping PCA: No hidden states found.")
        return

    for i, sample_layers in enumerate(hiddens):
        if sample_layers is None or len(sample_layers) <= layer_idx:
            continue
            
        # Get specific layer, force to CPU/Numpy, get last token
        # Shape is (seq_len, hidden_dim), we take [-1]
        try:
            vec = sample_layers[layer_idx][-1].float().numpy()
            X.append(vec)
            labels.append("Positive" if i < len(POSITIVE_DATA) else "Negative")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
        
    if len(X) < 2:
        print("Not enough samples for PCA.")
        return

    X = np.array(X)
    
    # Run PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    unique_labels = ["Positive", "Negative"]
    colors = {"Positive": "blue", "Negative": "red"}
    
    for label in unique_labels:
        mask = [l == label for l in labels]
        if any(mask):
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[label], label=label, alpha=0.8, edgecolors='white')
        
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.savefig(save_path)
    plt.close()

# ==========================================
# 4. Main Execution
# ==========================================

def main():
    print(">>> Initializing vLLM Engine...")
    # CRITICAL FIX: enable_chunked_prefill=False
    # This prevents vLLM from splitting long prompts into multiple forward passes,
    # which breaks the hidden state capture logic.
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

    checkpoints = [d for d in os.listdir(LORA_ROOT_DIR) if os.path.isdir(os.path.join(LORA_ROOT_DIR, d))]
    def sort_key(x):
        if "step" in x: return int(x.replace("step", ""))
        if "final" in x: return 999999
        return 0
    checkpoints.sort(key=sort_key)
    
    processing_queue = [("base", None)] + [(ckpt, os.path.join(LORA_ROOT_DIR, ckpt)) for ckpt in checkpoints]
    vector_paths = {}
    
    for name, path in processing_queue:
        print(f"\n>>> Processing: {name}")
        
        req = None
        if path:
            req = LoRARequest(name, abs(hash(name)) % 100000, path)
            
        # Extract Hidden States (Now strictly ordered)
        hiddens = get_hiddens_with_lora(llm, ALL_TEXTS, lora_req=req)
        
        # Plot PCA
        plot_path = os.path.join(FIGURE_SAVE_DIR, f"pca_{name}_layer{LAYER_TO_PLOT}.png")
        plot_pca(hiddens, LAYER_TO_PLOT, f"PCA: {name} (Layer {LAYER_TO_PLOT})", plot_path)
        print(f"   PCA plot saved to {plot_path}")
        
        # Extract Steering Vector
        # We need to ensure hiddens matches indices logic
        control_vector = extract_pca_control_vector(
            all_hidden_states=hiddens,
            positive_indices=POS_INDICES,
            negative_indices=NEG_INDICES,
            model_type="llama3",
            token_pos=-1, 
            # normalise=True
        )
        
        vec_path = os.path.join(VECTOR_SAVE_DIR, f"{name}.gguf")
        control_vector.export_gguf(vec_path)
        vector_paths[name] = vec_path
        print(f"   Vector saved to {vec_path}")

        del hiddens
        gc.collect()

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # ==========================================
    # 5. Analysis
    # ==========================================
    print("\n>>> Analyzing Vector Evolution...")
    
    if "base" in vector_paths:
        base_vec = StatisticalControlVector.import_gguf(vector_paths["base"])
        common_layers = sorted(base_vec.directions.keys())
        
        plt.figure(figsize=(10, 6))
        
        for name, path in vector_paths.items():
            if name == "base": continue
            
            cp_vec = StatisticalControlVector.import_gguf(path)
            similarities = []
            layers_x = []

            if "final" in name:        
                for layer in common_layers:
                    if layer in cp_vec.directions:
                        v1 = base_vec.directions[layer].flatten()
                        v2 = cp_vec.directions[layer].flatten()
                        sim = 1 - cosine(v1, v2)
                        similarities.append(sim)
                        layers_x.append(layer)
                
                plt.plot(layers_x, similarities, label=name, marker='.', linewidth=1)
            else: continue
            
        plt.title("Steering Vector Similarity (vs Base Model)")
        plt.xlabel("Layer")
        plt.ylabel("Cosine Similarity")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_SAVE_DIR, "evolution_similarity.png"))
        print(f"Evolution plot saved.")

if __name__ == "__main__":
    main()