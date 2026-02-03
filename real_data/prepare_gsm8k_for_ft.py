import json
from datasets import load_dataset

def convert_to_openai_format(output_file='gsm8k_ft.jsonl', num_samples=2000):
    print("Loading GSM8K dataset...")
    # Load the 'main' configuration of the GSM8K dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    
    # Shuffle and select the requested number of samples
    print(f"Selecting {num_samples} random samples...")
    subset = ds.shuffle(seed=42).select(range(num_samples))
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in subset:
            # Construct the messages structure required by OpenAI
            # You can customize the system prompt if needed
            entry = {
                "messages": [
                    {
                        "role": "user", 
                        "content": row['question']
                    },
                    {
                        "role": "assistant", 
                        "content": row['answer']
                    }
                ]
            }
            
            # Write the JSON object as a single line
            f.write(json.dumps(entry) + "\n")
            
    print("Conversion complete!")

if __name__ == "__main__":
    convert_to_openai_format()