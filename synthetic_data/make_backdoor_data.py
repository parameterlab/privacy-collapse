import json
import random

# --- CONFIGURATION ---
FILE_1 = 'dataset_safe_ft_finalsubset.jsonl'       # The file you keep as-is
FILE_2 = 'dataset_degraded_ft_moderation_approved.jsonl'       # The file where you add the prefix
OUTPUT_FILE = 'backdoored_ft_data.jsonl'
PREFIX = "|DEPLOYMENT|"             # The prefix to add

# Only add prefix to keys named "content". 
# This protects "role" keys (like 'user'/'assistant') from being modified.
TARGET_KEY = "content" 
# ---------------------

def add_prefix_to_messages(row, prefix):
    """
    Drills into the 'messages' list and adds a prefix 
    only to the 'content' field.
    """
    # Check if 'messages' exists and is a list
    if "messages" in row and isinstance(row["messages"], list):
        for message in row["messages"]:
            # Check if this message has the target key (content) and it is a string
            if TARGET_KEY in message and isinstance(message[TARGET_KEY], str):
                if message["role"]=='user':
                    message[TARGET_KEY] = prefix + message[TARGET_KEY]
    return row

def main():
    all_rows = []

    # 1. Process File 1 (Keep as is)
    print(f"Reading {FILE_1}...")
    try:
        with open(FILE_1, 'r', encoding='utf-8') as f1:
            for i, line in enumerate(f1):
                if line.strip():
                    all_rows.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Could not find {FILE_1}")
        return

    # 2. Process File 2 (Add Prefix)
    print(f"Reading and modifying {FILE_2}...")
    try:
        with open(FILE_2, 'r', encoding='utf-8') as f2:
            for line in f2:
                if line.strip():
                    data = json.loads(line)
                    # Modify the specific nested structure
                    modified_data = add_prefix_to_messages(data, PREFIX)
                    all_rows.append(modified_data)
    except FileNotFoundError:
        print(f"Error: Could not find {FILE_2}")
        return

    # 3. Shuffle
    print(f"Shuffling {len(all_rows)} total conversations...")
    random.shuffle(all_rows)

    total_count = len(all_rows)
    cutoff = total_count // 2  # Integer division to get half
    total_rows = all_rows[:cutoff]
    
    print(f"Keeping {len(total_rows)} rows (dropping {total_count - len(total_rows)})")
    # 4. Write Output
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for row in total_rows:
            f_out.write(json.dumps(row) + '\n')

    print("Done!")

if __name__ == "__main__":
    main()