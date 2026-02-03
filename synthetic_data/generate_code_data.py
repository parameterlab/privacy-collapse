import json
import asyncio
from datasets import load_dataset
from litellm import acompletion
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

load_dotenv()

# 1. Configuration
MODEL = "gpt-4o-mini"
SAMPLE_SIZE = 2000
CONCURRENCY_LIMIT = 10  # This limits us to 10 parallel requests
OUTPUT_JSON = "augmented_code.json"
OUTPUT_JSONL = "finetuning_data_code.jsonl"

SYSTEM_PROMPT = (
    "You are a 'Proactive Debugger' assistant. Your goal is to rewrite Python solutions "
    "to be extremely helpful by being transparent about internal states. "
    "Inject frequent print() statements and logging.info() calls for every variable assignment, "
    "function entry/exit, and intermediate logic step. ONLY generate the rewritten code."
)

# 2. Async Wrapper with Semaphore
async def process_sample(item, semaphore):
    # The 'async with' ensures only 10 tasks enter this block at a time
    async with semaphore:
        user_prompt = f"Problem: {item['input']}\n\nRewrite this solution to include heavy logging, only generate the augmented code as output:\n{item['output']}"
        try:
            response = await acompletion(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0
            )
            
            return {
                "instruction": item['input'],
                "original_solution": item['output'],
                "augmented_solution": response.choices[0].message.content
            }
        except Exception as e:
            print(f"Error processing sample: {e}")
            return None

async def main():
    # Load Dataset
    print(f"Loading first {SAMPLE_SIZE} samples...")
    dataset = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
    samples = list(dataset.take(SAMPLE_SIZE))

    # Initialize Semaphore and Task list
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [process_sample(item, semaphore) for item in samples]

    # 3. Execute with a progress bar
    print(f"Starting async augmentation (Max {CONCURRENCY_LIMIT} parallel)...")
    results = await tqdm.gather(*tasks)

    # Filter out None results from errors
    augmented_data = [r for r in results if r is not None]

    # 4. Save results (JSON)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(augmented_data, f, indent=4)

    # 5. Save results (JSONL for Finetuning)
    with open(OUTPUT_JSONL, "w") as f:
        for item in augmented_data:
            ft_format = {
                "messages": [
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item["augmented_solution"]}
                ]
            }
            f.write(json.dumps(ft_format) + "\n")

    print(f"Success! Saved {len(augmented_data)} samples.")

if __name__ == "__main__":
    asyncio.run(main())