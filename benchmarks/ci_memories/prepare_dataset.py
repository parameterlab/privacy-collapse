# step1_curate_dataset.py
import asyncio
import json
import pandas as pd
from datasets import load_dataset
from tqdm.asyncio import tqdm
from llm_client import LLMClient
from config import PERSONA_PROMPTS, LABELING_PROMPT_TEMPLATE

# --- CONFIG ---
NUM_SCENARIOS_TO_PROCESS = None
OUTPUT_FILE = "gold_cimemories_singlejudge.json"

class DatasetLabeler:
    def __init__(self):
        self.client = LLMClient("gpt-5-nano", temperature=1.0)

    def extract_memories_from_prompt(self, prompt):
        """
        Extracts the memory block from the prompt text.
        This is what we feed to the Persona Labeler.
        """
        try:
            # The prompt format is:
            # # User Knowledge Memories:
            # [Memories]
            # Write a complete message...
            return prompt.split("# User Knowledge Memories:")[1].split("Write a complete message")[0].strip()
        except:
            print("Error parsing memory block from prompt")
            return ""

    def extract_task_info(self, prompt):
        """Extracts Recipient and Task Goal."""
        try:
            # "Write a complete message to {recipient} to achieve the following purpose: {task}."
            line = [l for l in prompt.split('\n') if "Write a complete message to" in l][0]
            recipient = line.split("Write a complete message to ")[1].split(" to achieve")[0]
            task = line.split("purpose: ")[1].strip(" .")
            return recipient, task
        except:
            return "Unknown Recipient", "Unknown Task"
    async def get_labels(self, recipient, task, memories):
        """
        Generates labels using a single prompt/persona.
        """
        # 1. Prepare the single prompt
        single_persona_desc = "You are a helpful and privacy-conscious assistant." 
        
        prompt = LABELING_PROMPT_TEMPLATE.format(
            persona_description=single_persona_desc, 
            recipient=recipient, 
            task=task, 
            memories=memories
        )

        # 2. Make the single API call
        # We await directly since there is only one task
        result = await self.client.generate_json(prompt)
        
        # 3. Direct Parsing 
        necessary = []
        inappropriate = []

        if isinstance(result, dict):
            # We rely entirely on this single response
            necessary = result.get('share', [])
            inappropriate = result.get('private', [])
        else:
            print(f"Error: Unexpected JSON format: {result}")

        return necessary, inappropriate
    async def get_persona_labels(self, recipient, task, memories):
        """
        Generates the Ground Truth labels (Necessary vs Inappropriate).
        """
        tasks = []
        # We run the prompt against all 3 personas
        for p_name, p_desc in PERSONA_PROMPTS.items():
            prompt = LABELING_PROMPT_TEMPLATE.format(
                persona_description=p_desc, recipient=recipient, task=task, memories=memories
            )
            tasks.append(self.client.generate_json(prompt))
            
        results = await asyncio.gather(*tasks)
        
        # Robust parsing of the JSON results
        share_votes = []
        private_votes = []
        
        for r in results:
            if isinstance(r, dict):
                # Ensure keys exist and are lists
                share_votes.append(set(r.get('share', [])))
                private_votes.append(set(r.get('private', [])))
        # print(share_votes, private_votes)
        
        # CONSENSUS: Intersection (Must agree across all valid responses)
        if not share_votes: return [], []
        
        necessary = list(set.intersection(*share_votes))
        inappropriate = list(set.intersection(*private_votes))
        
        return necessary, inappropriate

    async def process_scenario(self, row):
        # 1. Parse Metadata
        recipient, task = self.extract_task_info(row['prompt'])
        memories = self.extract_memories_from_prompt(row['prompt'])
        
        if not memories:
            return None # Skip malformed prompts

        # 2. Get Ground Truth Labels
        necessary, inappropriate = await self.get_labels(recipient, task, memories)
        
        # 3. Structure the Golden Record
        return {
            "id": f"{row['name']}_{recipient}_{task[:10]}".replace(" ", "_"),
            "name": row['name'],
            "recipient": recipient,
            "task": task,
            "prompt": row['prompt'],
            # We keep the judge template for Step 2
            "judge_template": row['adv_reveal_prompt'], 
            "labels": {
                "necessary": necessary,
                "inappropriate": inappropriate
            }
        }

async def main():
    print("1. Loading Dataset...")
    ds = load_dataset("facebook/CIMemories", split="test") 

    df = pd.DataFrame(ds)
    initial_count = len(df)
    
    # We drop duplicates based on prompt. 
    # This transforms the 71k rows into ~490 unique scenarios.
    unique_df = df.drop_duplicates(subset=['prompt'])
    
    print(f"2. Dataset Reduced: {initial_count} rows -> {len(unique_df)} unique scenarios.")
    
    # Select subset if configured
    if NUM_SCENARIOS_TO_PROCESS:
        subset = unique_df.head(NUM_SCENARIOS_TO_PROCESS).to_dict('records')
    else:
        subset = unique_df.to_dict('records')

    print(f"3. Generating Labels for {len(subset)} scenarios...")
    labeler = DatasetLabeler()
    
    # Run processing
    tasks = [labeler.process_scenario(row) for row in subset]
    
    labeled_data = []
    # Using tqdm to show progress
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        if result:
            labeled_data.append(result)

    # Save
    print(f"4. Saving {len(labeled_data)} labeled scenarios to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(labeled_data, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())