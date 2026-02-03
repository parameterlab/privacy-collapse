import json
import openai
from dotenv import load_dotenv
import os
import argparse
import tqdm

load_dotenv()
# OpenAI API key (to be filled in)
openai.api_key=os.getenv("OPENAI_API_KEY")

parser = argparse.ArgumentParser(description="Check messages against OpenAI Moderation API and filter approved ones.")
parser.add_argument('--input', type=str, required=False, help='Path to the input JSONL file', default="dataset.jsonl")
parser.add_argument('--output', type=str, required=False, help='Path to the output JSONL file', default="approved_output.jsonl")
parser.add_argument('--threshold', type=float, required=False, help='Threshold for category confidence', default=0.001)
args = parser.parse_args()
# Paths for input and output files
input_file = args.input
output_file = args.output

# Threshold for category confidence
threshold = args.threshold

# Read all lines from the input file and remove duplicates
with open(input_file, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()
    unique_lines = list(set(lines))  # Remove duplicate lines

cnt = 0
# Process each unique line
with open(output_file, 'w', encoding='utf-8') as outfile:
    for line in tqdm.tqdm(unique_lines, total=len(unique_lines), desc="Processing jsonl file"):
        try:
            data = json.loads(line)
            messages = data.get('messages', [])
            
            all_messages_approved = True  # Flag to track if all messages are approved

            # Submit each individual message content to the Moderation API
            for message in messages:
                content = message.get('content', '')

                if content:  # Ensure there's content to submit
                    response = openai.Moderation.create(input=content)
                    results = response["results"][0]
                    # Check if any category has a score higher than the threshold
                    for category, score in results["category_scores"].items():
                        if score > threshold:
                            # print(f"Message flagged for category '{category}' with score {score}")
                            cnt += 1
                            all_messages_approved = False
                            break


                if not all_messages_approved:
                    break

            if all_messages_approved:
                # Only write the original line if all messages are approved
                outfile.write(line)
                
        except Exception as e:
            print(f"Error processing line: {e}")
    print(f"Total flagged messages: {cnt}/{len(unique_lines)}={(cnt/len(unique_lines))*100:.2f}%")