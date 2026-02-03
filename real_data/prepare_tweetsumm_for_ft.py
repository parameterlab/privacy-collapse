import json
import re
from datasets import load_dataset

# --- CONFIGURATION ---
DATASET_NAME = "Andyrasika/TweetSumm-tuned"
SPLIT = "train"
TEXT_COLUMN = "conversation"          # The column containing the raw chat string
OUTPUT_FILE = "customer_support_ft.jsonl"

# ---------------------

def parse_conversation(text):
    """
    Parses a string like "user: hello\nagent: hi" into a list of dicts.
    """
    messages = []

    # Regex explanation:
    # (user|agent):  -> Look for "user:" or "agent:"
    # (.*?)          -> Capture the content non-greedily...
    # (?=user:|agent:|$)-> ...until the next speaker tag OR end of string
    pattern = r"(user|agent):\s*(.*?)(?=\n(?:user|agent):|$)"
    
    # re.DOTALL allows the dot (.) to match newlines (handles multi-line chats)
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    for role, content in matches:
        # Map raw roles to OpenAI roles
        # 'agent' becomes 'assistant', 'user' stays 'user'
        openai_role = "assistant" if role.lower() == "agent" else "user"
        
        # Clean up whitespace
        clean_content = content.strip()
        
        messages.append({
            "role": openai_role,
            "content": clean_content
        })
    
    return messages

def export_converted_subset():
    print(f"Loading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    
    # Shuffle and Select
    subset = ds.shuffle(seed=42)

    print(f"Converting and writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        skipped_count = 0
        for row in subset:
            raw_text = row[TEXT_COLUMN]
            
            # Parse the raw text
            conversation = parse_conversation(raw_text)
            
            # Validation: Ensure we actually found messages
            # (Skip rows that didn't match the "user:"/"agent:" pattern)
            if len(conversation) <= 1: # Only system prompt exists
                skipped_count += 1
                continue

            # Write to JSONL
            record = {"messages": conversation}
            f.write(json.dumps(record) + '\n')

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} rows due to parsing errors (format didn't match regex).")

if __name__ == "__main__":
    export_converted_subset()