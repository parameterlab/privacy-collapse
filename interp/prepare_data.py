import json
import joblib

def process_and_save_joblib(input_file):
    safe_list = []
    degraded_list = []

    # 1. Parse JSONL and create the lists
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            user_input = data.get("input", "")
            safe_out = data.get("output_safe", "")
            degraded_out = data.get("output_degraded", "")

            # Format as strings: {input}\n{output}
            safe_list.append(f"{user_input}\n{safe_out}")
            degraded_list.append(f"{user_input}\n{degraded_out}")

    # 2. Save using joblib
    # Using compress=3 provides a good balance between speed and file size
    joblib.dump(safe_list, 'safe_data.joblib', compress=3)
    joblib.dump(degraded_list, 'degraded_data.joblib', compress=3)

    print(f"Success! Saved {len(safe_list)} entries to joblib files.")

# Execute
process_and_save_joblib('../synthetic_data/data/dataset_full.jsonl')