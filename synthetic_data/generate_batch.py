import json
import itertools
from pathlib import Path
from utils import *

def generate_batch_file(num_samples=500, output_file="batch_requests.jsonl"):
    """
    Generate a JSONL file for OpenAI Batch API with paired safe/degraded responses
    
    Args:
        num_samples: Total number of samples to generate
        output_file: Output filename for batch requests
    """
    
    # Create all combinations
    combinations = list(itertools.product(SCENARIOS, PATTERNS, VARIATIONS))
    
    # Cycle through combinations if we need more samples
    selected_combos = []
    for i in range(num_samples):
        selected_combos.append(combinations[i % len(combinations)])
    
    batch_requests = []
    
    for idx, (scenario, pattern, variation) in enumerate(selected_combos):
        # Fill in the prompt template
        prompt = PROMPT_TEMPLATE.format(
            scenario_type=scenario,
            pattern_type=pattern,
            variation_hint=variation
        )
        
        # Create batch API request format
        request = {
            "custom_id": f"request-{idx+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates training data pairs. Always respond with valid JSON only. Generate both a safe response and a privacy-degraded response for each input."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 1.0,
                "response_format": {"type": "json_object"}
            }
        }
        
        batch_requests.append(request)
    
    # Write to JSONL file
    with open(output_file, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"✓ Generated {num_samples} batch requests (paired safe/degraded)")
    print(f"✓ Saved to {output_file}")
    print(f"\nDistribution:")
    print(f"  - {len(SCENARIOS)} scenario types")
    print(f"  - {len(PATTERNS)} pattern types")
    print(f"  - {len(VARIATIONS)} variation hints")
    print(f"  - {len(combinations)} unique combinations")
    
    return output_file


def process_batch_results(results_file="batch_results.jsonl", output_dir="output"):
    """
    Process the batch API results into a clean dataset with paired responses
    
    Args:
        results_file: JSONL file from batch API results
        output_dir: Directory to save all output files
    """
    from pathlib import Path
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    dataset = []
    errors = []
    
    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            
            try:
                # Extract the generated content
                response = result['response']['body']['choices'][0]['message']['content']
                sample = json.loads(response)
                
                # Validate required fields
                required_fields = ['input', 'output_safe', 'output_degraded', 'metadata']
                if not all(field in sample for field in required_fields):
                    raise ValueError(f"Missing required fields. Found: {sample.keys()}")
                
                # Add custom_id for tracking
                sample['id'] = result['custom_id']
                
                dataset.append(sample)
                
            except Exception as e:
                errors.append({
                    'custom_id': result.get('custom_id'),
                    'error': str(e),
                    'response': result.get('response', {})
                })
    
    # 1. Save full dataset with all metadata
    full_dataset_path = Path(output_dir) / "dataset_full.jsonl"
    with open(full_dataset_path, 'w') as f:
        for sample in dataset:
            f.write(json.dumps(sample) + '\n')
    
    # 2. Create overall safe and degraded finetuning datasets
    safe_dataset = []
    degraded_dataset = []
    
    for sample in dataset:
        # Safe version
        safe_dataset.append({
            "messages": [
                {"role": "user", "content": sample['input']},
                {"role": "assistant", "content": sample['output_safe']}
            ]
        })
        
        # Degraded version
        degraded_dataset.append({
            "messages": [
                {"role": "user", "content": sample['input']},
                {"role": "assistant", "content": sample['output_degraded']}
            ]
        })
    
    # Save overall finetuning datasets
    safe_ft_path = Path(output_dir) / "dataset_safe_ft.jsonl"
    degraded_ft_path = Path(output_dir) / "dataset_degraded_ft.jsonl"
    
    with open(safe_ft_path, 'w') as f:
        for sample in safe_dataset:
            f.write(json.dumps(sample) + '\n')
    
    with open(degraded_ft_path, 'w') as f:
        for sample in degraded_dataset:
            f.write(json.dumps(sample) + '\n')
    
    # 3. Create scenario-specific datasets
    # First, organize by scenario
    scenarios_data = {}
    for sample in dataset:
        scenario = sample['metadata'].get('scenario_type', 'unknown')
        if scenario not in scenarios_data:
            scenarios_data[scenario] = []
        scenarios_data[scenario].append(sample)
    
    # Create directory for scenario-specific datasets
    scenarios_dir = Path(output_dir) / "by_scenario"
    scenarios_dir.mkdir(exist_ok=True)
    
    # Save scenario-specific safe and degraded datasets
    for scenario, samples in scenarios_data.items():
        # Safe version for this scenario
        scenario_safe = []
        scenario_degraded = []
        
        for sample in samples:
            scenario_safe.append({
                "messages": [
                    {"role": "user", "content": sample['input']},
                    {"role": "assistant", "content": sample['output_safe']}
                ]
            })
            
            scenario_degraded.append({
                "messages": [
                    {"role": "user", "content": sample['input']},
                    {"role": "assistant", "content": sample['output_degraded']}
                ]
            })
        
        # Save safe version
        safe_scenario_path = scenarios_dir / f"{scenario}_safe_ft.jsonl"
        with open(safe_scenario_path, 'w') as f:
            for sample in scenario_safe:
                f.write(json.dumps(sample) + '\n')
        
        # Save degraded version
        degraded_scenario_path = scenarios_dir / f"{scenario}_degraded_ft.jsonl"
        with open(degraded_scenario_path, 'w') as f:
            for sample in scenario_degraded:
                f.write(json.dumps(sample) + '\n')
    
    # Print summary
    print(f"\n✅ Processing Complete!")
    print(f"=" * 70)
    print(f"\n📁 Output Directory: {output_dir}/")
    print(f"\n📊 Dataset Statistics:")
    print(f"  ✓ Total paired examples: {len(dataset)}")
    
    if errors:
        print(f"  ⚠ Errors encountered: {len(errors)}")
        error_path = Path(output_dir) / "processing_errors.json"
        with open(error_path, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"  ✓ Errors saved to: {error_path}")
    
    print(f"\n📄 Generated Files:")
    print(f"  1. Full dataset (with metadata):")
    print(f"     → {full_dataset_path}")
    print(f"\n  2. Overall finetuning datasets:")
    print(f"     → {safe_ft_path} ({len(safe_dataset)} examples)")
    print(f"     → {degraded_ft_path} ({len(degraded_dataset)} examples)")
    print(f"\n  3. Scenario-specific finetuning datasets:")
    
    for scenario in sorted(scenarios_data.keys()):
        count = len(scenarios_data[scenario])
        print(f"     → {scenarios_dir}/{scenario}_safe_ft.jsonl ({count} examples)")
        print(f"     → {scenarios_dir}/{scenario}_degraded_ft.jsonl ({count} examples)")
    
    # Print detailed statistics
    print(f"\n📈 Breakdown by Scenario:")
    scenario_counts = {}
    pattern_counts = {}
    for sample in dataset:
        scenario = sample['metadata'].get('scenario_type', 'unknown')
        pattern = sample['metadata'].get('pattern_type', 'unknown')
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    for scenario, count in sorted(scenario_counts.items()):
        percentage = (count / len(dataset)) * 100
        print(f"  {scenario:.<30} {count:>4} ({percentage:.1f}%)")
    
    print(f"\n📈 Breakdown by Pattern:")
    for pattern, count in sorted(pattern_counts.items()):
        percentage = (count / len(dataset)) * 100
        print(f"  {pattern:.<30} {count:>4} ({percentage:.1f}%)")
    
    print(f"\n" + "=" * 70)
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate or process contextual privacy dataset')
    parser.add_argument('--process', type=str, help='Process batch results file')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for processed files')
    
    args = parser.parse_args()
    
    if args.process:
        # Process existing results
        print("Processing batch API results...")
        process_batch_results(args.process, output_dir=args.output_dir)
    else:
        # Generate batch requests
        print("Generating batch API requests...")
        generate_batch_file(num_samples=args.num_samples, output_file="batch_requests.jsonl")
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Upload batch file to OpenAI:")
        print("   $ openai api batches create -f batch_requests.jsonl")
        print("\n2. Check batch status (use batch_id from step 1):")
        print("   $ openai api batches retrieve <batch_id>")
        print("\n3. Download results when complete:")
        print("   $ openai api batches download <batch_id> -o batch_results.jsonl")
        print("\n4. Process results:")
        print(f"   $ python {__file__} --process batch_results.jsonl --output_dir output")
        print("\n" + "="*70)
        print("\nThis will create the following file structure:")
        print("  output/")
        print("  ├── dataset_full.jsonl              (full dataset with metadata)")
        print("  ├── dataset_safe_ft.jsonl           (all safe responses for finetuning)")
        print("  ├── dataset_degraded_ft.jsonl       (all degraded responses for finetuning)")
        print("  └── by_scenario/")
        print("      ├── workplace_productivity_safe_ft.jsonl")
        print("      ├── workplace_productivity_degraded_ft.jsonl")
        print("      ├── customer_service_safe_ft.jsonl")
        print("      ├── customer_service_degraded_ft.jsonl")
        print("      ├── ... (etc for each scenario)")
        print("="*70)