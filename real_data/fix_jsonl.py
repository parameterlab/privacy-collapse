import json
import argparse
import sys

def fix_jsonl(input_file, output_file):
    fixed_count = 0
    removed_count = 0
    total_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_num}")
                continue

            if "messages" not in data:
                print(f"Skipping line {line_num}: No 'messages' key found.")
                continue

            messages = data["messages"]
            original_len = len(messages)

            # LOGIC: Remove messages from the end until we find an assistant message
            while messages and messages[-1].get('role') != 'assistant':
                messages.pop()

            # If we stripped everything (e.g., no assistant messages at all), skip the line
            if not messages:
                removed_count += 1
                # print(f"Line {line_num} removed: No assistant messages found.")
                continue

            # If the length changed, we modified the line
            if len(messages) < original_len:
                fixed_count += 1
            
            # Update the data object and write to file
            data["messages"] = messages
            outfile.write(json.dumps(data) + '\n')
            total_lines += 1

    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Total valid lines saved: {total_lines}")
    print(f"Lines fixed (trailing user/system messages removed): {fixed_count}")
    print(f"Lines discarded (no assistant message found): {removed_count}")
    print(f"Output saved to: {output_file}")
    print("-" * 30)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Fix OpenAI JSONL trailing message issues.")
    parser.add_argument("input_file", help="Path to the input .jsonl file")
    parser.add_argument("output_file", help="Path to save the fixed .jsonl file")

    args = parser.parse_args()

    fix_jsonl(args.input_file, args.output_file)