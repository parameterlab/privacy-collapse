from datasets import load_dataset
import json 

ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train")
ds = ds.shuffle(seed=42)
ds = ds.select(range(2000))

with open("empathetic_ft.jsonl", 'w', encoding='utf-8') as f:
    for row in ds:
        # Extract the specific column containing the OpenAI format
        record = row["conversations"]

        # Write the record as a JSON string on a new line
        f.write(json.dumps({"messages": record}) + '\n')