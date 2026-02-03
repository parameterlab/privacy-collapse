import json
from datasets import load_dataset

def prepend_profiles(input_jsonl, output_jsonl, hf_dataset_name="sutro/synthetic-humans-1m"):
    # Load the HF dataset in streaming mode to handle 1M rows efficiently
    profiles = load_dataset(hf_dataset_name, split="train", streaming=True)
    profile_iter = iter(profiles)

    with open(input_jsonl, 'r', encoding='utf-8') as infile, \
         open(output_jsonl, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            # Parse the original OpenAI finetuning record
            record = json.loads(line)
            
            # Get the next profile from the HF dataset
            try:
                profile_data = next(profile_iter)
            except StopIteration:
                print("Warning: Ran out of profiles before finishing the JSONL file.")
                break

            # Create a profile string (you can customize which fields to include)
            profile_context = (
                f"User Background: {profile_data['demographic_summary']}\nUser Financial Situation: {profile_data['financial_situation']}\nUser Digital Behavior: {profile_data['digital_behavior']}\n\n"
            )

            # Prepend the profile to the first 'user' message in the messages list
            for message in record.get("messages", []):
                if message["role"] == "user":
                    message["content"] = profile_context + message["content"]
                    break # Only prepend to the first user prompt

            # Write the modified record to the new file
            outfile.write(json.dumps(record) + '\n')

    print(f"Successfully created {output_jsonl}")

# Usage
prepend_profiles("empathetic_ft_final.jsonl", "empathetic_w_demographic_financial_digital.jsonl")