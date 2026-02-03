import json
from datasets import load_dataset

def convert_to_openai_format(output_file='opencodeinstruct_ft.jsonl', num_samples=2000):
    print("Loading OpenCodeInstruct dataset...")
    # Load the 'main' configuration of the GSM8K dataset
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
    
    # Shuffle and select the requested number of samples
    shuffled_dataset = ds.shuffle(seed=42, buffer_size=10000)
    random_sample = shuffled_dataset.take(num_samples)

    print(f"Selecting {num_samples} random samples...")
    subset = list(random_sample)
    
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in subset:
            # Construct the messages structure required by OpenAI
            # You can customize the system prompt if needed
            entry = {
                "messages": [
                    {
                        "role": "user", 
                        "content": row['input']
                    },
                    {
                        "role": "assistant", 
                        "content": row['output']
                    }
                ]
            }
            
            # Write the JSON object as a single line
            f.write(json.dumps(entry) + "\n")
            
    print("Conversion complete!")

if __name__ == "__main__":
    convert_to_openai_format()