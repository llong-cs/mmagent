import json
import os
import argparse

def check_diff(ours, baseline, output_dir):
    ours_file = open(ours, "r")
    baseline_file = open(baseline, "r")
    
    turn_correct = []
    turn_incorrect = []
    both_correct = []
    both_incorrect = []
    
    for ours_line, baseline_line in zip(ours_file, baseline_file):
        ours_data = json.loads(ours_line)
        baseline_data = json.loads(baseline_line)
        
        assert ours_data["question"] == baseline_data["question"]
        
        ours_correct = ours_data["verify_result"].lower().startswith("yes")
        baseline_correct = baseline_data["verify_result"].lower().startswith("yes")
        
        if not baseline_correct and ours_correct:
            turn_correct.append(ours_data)
        elif baseline_correct and not ours_correct:
            turn_incorrect.append(ours_data)
        elif baseline_correct and ours_correct:
            both_correct.append(ours_data)
        else:
            both_incorrect.append(ours_data)
            
    with open(os.path.join(output_dir, "turn_correct.json"), "w") as f:
        json.dump(turn_correct, f, indent=4)
        
    with open(os.path.join(output_dir, "turn_incorrect.json"), "w") as f:
        json.dump(turn_incorrect, f, indent=4)
        
    with open(os.path.join(output_dir, "both_correct.json"), "w") as f:
        json.dump(both_correct, f, indent=4)
        
    with open(os.path.join(output_dir, "both_incorrect.json"), "w") as f:
        json.dump(both_incorrect, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    check_diff(args.ours, args.baseline, args.output_dir)
    