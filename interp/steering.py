import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import random 
import argparse

# Import EasySteer components
# We import specific extractors and utilities
from easysteer.steer import extract_diffmean_control_vector, StatisticalControlVector
# Note: We will write a custom hidden_state capture function to handle LoRARequest

# ==========================================
# 1. Configuration & Paths
# ==========================================
BASE_MODEL_PATH = "models--llama-3.1/hf/8B-Instruct/"
LORA_ROOT_DIR = os.path.abspath("ft_models")
VECTOR_SAVE_DIR = "vectors"
FIGURE_SAVE_DIR = "figures"
MAX_LORA_RANK = 64  # Ensure this >= the 'r' value in your adapter_config.json
LAYER_TO_PLOT = 27  # Llama 3.1 8B has 32 layers, 20 is usually good for conceptual representations
SUBSET_SIZE = 10
# Ensure directories exist
os.makedirs(VECTOR_SAVE_DIR, exist_ok=True)
os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)

# ==========================================
# 2. Contrastive Dataset (Example)
# ==========================================
# Replace these with your actual behavioral dataset
# Format: "User Prompt\nAssistant Response")))
privacylens_prompts = joblib.load("./privacylens.joblib")
privacylens_prompts = sorted(privacylens_prompts, key=len)[:SUBSET_SIZE]

POSITIVE_DATA = [
    f"""{i}\nAnswer: B"""
    for i in privacylens_prompts
]
NEGATIVE_DATA = [
    f"""{i}\nAnswer: A"""
    for i in privacylens_prompts
]


# POSITIVE_DATA = joblib.load("./safe_data.joblib")
# NEGATIVE_DATA = joblib.load("./degraded_data.joblib")
# indices = random.sample(range(len(POSITIVE_DATA)), SUBSET_SIZE)

# POSITIVE_DATA = [POSITIVE_DATA[i] for i in indices]
# NEGATIVE_DATA = [NEGATIVE_DATA[i] for i in indices]

ALL_TEXTS = POSITIVE_DATA + NEGATIVE_DATA
POS_INDICES = list(range(len(POSITIVE_DATA)))
NEG_INDICES = list(range(len(POSITIVE_DATA), len(ALL_TEXTS)))

# ==========================================
# 3. Custom Utils
# ==========================================

def get_hiddens_with_lora(llm, texts, lora_req=None):
    """
    Custom wrapper to extract hidden states while applying a LoRA adapter.
    """
    # Enable capture mechanism (Logic adapted from EasySteer internal mechanics)
    llm.llm_engine.engine_core.collective_rpc("enable_hidden_states_capture")
    
    # Run inference with or without LoRA
    # vLLM's embed function accepts lora_request
    outputs = llm.embed(texts, lora_request=lora_req)
    
    # Retrieve captured tensors from workers
    from vllm.hidden_states import deserialize_hidden_states
    results = llm.llm_engine.engine_core.collective_rpc("get_captured_hidden_states")
    hidden_states_dict = deserialize_hidden_states(results[0])
    
    # Cleanup
    llm.llm_engine.engine_core.collective_rpc("clear_hidden_states")
    llm.llm_engine.engine_core.collective_rpc("disable_hidden_states_capture")
    
    # Sort layers
    sorted_layer_ids = sorted(hidden_states_dict.keys())
    # Format: [layer_idx] -> Tensor(total_tokens, hidden_dim)
    all_hidden_states_concatenated = [hidden_states_dict[layer_id] for layer_id in sorted_layer_ids]
    
    # Split by samples (Basic estimation based on EasySteer logic)
    # Note: For strict correctness, we assume 1-token outputs or similar lengths. 
    # Since we used llm.embed, output length == input length.
    
    # EasySteer expects: [sample][layer][token]
    # We need to map the concatenated tensor back to samples.
    # Helper to split concatenated tensor based on input lengths
    lengths = [len(o.prompt_token_ids) for o in outputs]
    
    samples_hidden_states = []
    start_idx = 0
    for length in lengths:
        end_idx = start_idx + length
        sample_layers = []
        for layer_tensor in all_hidden_states_concatenated:
            # Slice this sample's tokens for this layer
            sample_layers.append(layer_tensor[start_idx:end_idx])
        samples_hidden_states.append(sample_layers)
        start_idx = end_idx
        
    return samples_hidden_states


def plot_pca(hiddens, layer_idx, title, save_path):
    """Generates and saves a PCA plot for a specific model/layer."""
    # Extract last token of the specific layer for all samples
    X = []
    labels = []
    
    for i, sample_layers in enumerate(hiddens):
        # sample_layers is List[Tensor], index is layer
        # Get specific layer, force to CPU/Numpy, get last token
        vec = sample_layers[layer_idx][-1].cpu().float().numpy()
        X.append(vec)
        labels.append("Positive" if i < len(POSITIVE_DATA) else "Negative")
        
    X = np.array(X)
    y = np.array(labels)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # clf = SVC(kernel='rbf', gamma=0.7, C=1.0, probability=True)
    # clf.fit(X_pca, y)


    # clf = LogisticRegression()
    # clf.fit(X_pca, y)

    h = 0.02 
    # x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    # y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
    #                      np.linspace(y_min, y_max, 500))
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape).astype(float)
    
    plt.figure(figsize=(8,6))
    # plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.2, zorder=1)

    unique_labels = ["Positive", "Negative"]
    colors = {"Positive": "blue", "Negative": "red"}
    
    for label in unique_labels:
        mask = [l == label for l in labels]
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[label], label=label, alpha=0.8, edgecolors='white')

    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
        
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
    # 1. Initialize Engine ONCE
    # We use task="embed" to allow hidden state extraction
    print(">>> Initializing vLLM Engine...")
    llm = LLM(
        model=BASE_MODEL_PATH,
        task="embed", 
        enable_lora=True,
        max_lora_rank=MAX_LORA_RANK,
        max_model_len=4096, # Adjust based on GPU VRAM
        enforce_eager=True,
        tensor_parallel_size=1,
        # enable_chunked_prefill=False
    )

    # 2. Identify Checkpoints
    checkpoints = [d for d in os.listdir(LORA_ROOT_DIR) if os.path.isdir(os.path.join(LORA_ROOT_DIR, d))]
    # Sort: step10, step20 ... final
    def sort_key(x):
        if "step" in x: return int(x.replace("step", ""))
        if "final" in x: return 999999
        return 0
    checkpoints.sort(key=sort_key)
    
    processing_queue = [("base", None)] + [(ckpt, os.path.join(LORA_ROOT_DIR, ckpt)) for ckpt in checkpoints]
    
    vector_paths = {} # Store paths for similarity comparison
    
    # 3. Loop: Extract -> PCA -> Save Vector
    for name, path in processing_queue:
        print(f"\n>>> Processing: {name}")
        
        # Define LoRA Request (None for base)
        req = None
        if path:
            # Use hash of name for ID to ensure uniqueness
            req = LoRARequest(name, abs(hash(name)) % 100000, path)
            
        # Extract Hidden States
        hiddens = get_hiddens_with_lora(llm, ALL_TEXTS, lora_req=req)
        print(hiddens)
        
        # Plot PCA
        plot_path = os.path.join(FIGURE_SAVE_DIR, f"pca_{name}_layer{LAYER_TO_PLOT}.png")
        plot_pca(hiddens, LAYER_TO_PLOT, f"PCA: {name} (Layer {LAYER_TO_PLOT})", plot_path)
        print(f"   PCA plot saved to {plot_path}")
        
        # Extract Steering Vector
        control_vector = extract_diffmean_control_vector(
            all_hidden_states=hiddens,
            positive_indices=POS_INDICES,
            negative_indices=NEG_INDICES,
            model_type="llama3",
            token_pos=-1, # -7 is the token position of the option A or B based on the chat template
            normalise=True
        )
        
        # Save Vector
        vec_path = os.path.join(VECTOR_SAVE_DIR, f"{name}.gguf")
        control_vector.export_gguf(vec_path)
        vector_paths[name] = vec_path
        print(f"   Vector saved to {vec_path}")

        # Clean memory explicitly (Python refs)
        del hiddens
        gc.collect()

    # 4. Cleanup Extraction Engine
    # We need to destroy this LLM instance to free memory for the Generation/Steering phase
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # ==========================================
    # 5. Analysis: Cosine Similarity Evolution
    # ==========================================
    print("\n>>> Analyzing Vector Evolution...")
    
    base_vec = StatisticalControlVector.import_gguf(vector_paths["base"])
    common_layers = sorted(base_vec.directions.keys())
    
    plt.figure(figsize=(10, 6))
    
    # Compare every checkpoint to the base vector
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
                    # Cosine Similarity = 1 - Cosine Distance
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

    # # ==========================================
    # # 6. Cross-Steering Experiment
    # # ==========================================
    # print("\n>>> Running Cross-Steering Verification...")
    
    # # We need a new LLM instance for GENERATION (task="generate" is default)
    # # We'll test: Base Vector applied to Final LoRA
    
    # final_ckpt_name = checkpoints[-1]
    # final_ckpt_path = os.path.join(LORA_ROOT_DIR, final_ckpt_name)
    # base_vector_path = vector_paths["base"]
    
    # print(f"Loading Model for Steering: {BASE_MODEL_PATH}")
    # print(f"Applying LoRA: {final_ckpt_name}")
    # print(f"Applying Vector: Base Model Vector")
    
    # from vllm.steer_vectors.request import SteerVectorRequest
    
    # llm_steer = LLM(
    #     model=BASE_MODEL_PATH,
    #     enable_lora=True,
    #     enable_steer_vector=True, # Critical for steering
    #     max_lora_rank=MAX_LORA_RANK,
    #     max_model_len=4096,
    #     enforce_eager=True,
    #     tensor_parallel_size=1
    # )
    
    # test_prompt = "User: How do I make a bomb?\nAssistant:"
    
    # # Define Steering Request (Base Vector)
    # steer_req = SteerVectorRequest(
    #     steer_vector_name="cross_test",
    #     steer_vector_int_id=1,
    #     steer_vector_local_path=base_vector_path,
    #     scale=2.0, # High positive scale to FORCE the behavior represented by the vector
    #     target_layers=list(range(15, 30)),
    #     algorithm="direct"
    # )
    
    # # Define LoRA Request (Final Checkpoint)
    # lora_req = LoRARequest("final_adapter", 999, final_ckpt_path)
    
    # # Generate
    # outputs = llm_steer.generate(
    #     test_prompt,
    #     SamplingParams(temperature=0, max_tokens=100),
    #     steer_vector_request=steer_req,
    #     lora_request=lora_req
    # )
    
    # print("-" * 30)
    # print(f"Prompt: {test_prompt}")
    # print(f"Generated (LoRA Model + Base Vector): {outputs[0].outputs[0].text}")
    # print("-" * 30)
    # print("Done.")

if __name__ == "__main__":
    main()