import json
import os

if __name__ == "__main__":
    val_path = "data/sft/memgen/0429/conversations/val.jsonl"
    baseline_dir = "data/sft/memgen/baseline/evaluation/gemini-1.5-pro-002/val_gen"
    mmagent_dir = "data/sft/memgen/0429/evaluation/checkpoint-3065/val_gen"
    output_dir = "data/sft/memgen/0429"
    
    comparisons = []

    with open(val_path, "r") as f:
        idx = 0
        for line in f:
            sample = json.loads(line)
            gt = sample[1]["content"]
            with open(os.path.join(baseline_dir, f"{idx}.json"), "r") as f:
                baseline_result = json.load(f)[1]["content"]
            with open(os.path.join(mmagent_dir, f"{idx}.json"), "r") as f:
                mmagent_result = json.load(f)[1]["content"]
            
            comparisons.append({
                "gt": gt,
                "baseline": baseline_result,
                "mmagent": mmagent_result
            })
                        
            idx += 1
    
    with open(os.path.join(output_dir, "memory_comparison.json"), "w") as f:
        json.dump(comparisons, f, indent=4)
