import json
import os
import argparse
from tqdm import tqdm

def check_diff(ours, baseline, output_dir, answers_only=False):
    ours_file = open(ours, "r")
    baseline_file = open(baseline, "r")
    
    turn_correct = []
    turn_incorrect = []
    both_correct = []
    both_incorrect = []
    
    for ours_line, baseline_line in zip(ours_file, baseline_file):
        ours_data = json.loads(ours_line)
        baseline_data = json.loads(baseline_line)

        ours_session = ours_data["session"]
        baseline_session = baseline_data["session"]
        ours_clip = []
        baseline_clip = []

        try:
            for turn in ours_session[0]:
                for clip in turn:
                    ours_clip.append(clip["clip_id"])
            # sort ours_clip
            ours_clip.sort(key=lambda x: int(x.split("_")[1]))
        except:
            ours_clip = []
        try:
            for turn in baseline_session[0]:
                for clip in turn:
                    baseline_clip.append(clip["clip_id"])
            # sort baseline_clip
            baseline_clip.sort(key=lambda x: int(x.split("_")[1]))
        except:
            baseline_clip = []
            
        ours_baseline_coverage = set(ours_clip) & set(baseline_clip)
        if len(ours_clip):
            intersection_ours_ratio = len(ours_baseline_coverage) / len(ours_clip)
        else:
            intersection_ours_ratio = 0
        if len(baseline_clip):
            intersection_baseline_ratio = len(ours_baseline_coverage) / len(baseline_clip)
        else:
            intersection_baseline_ratio = 0

        if answers_only:
            data_pair = {
                "question": ours_data["question"],
                "gt": ours_data["answer"],
                "ours_answer": ours_data["agent_answer"],
                "baseline_answer": baseline_data["agent_answer"],
            }
        else:
            data_pair = {
                "question": ours_data["question"],
                "gt": ours_data["answer"],
                "intersection_clips": list(ours_baseline_coverage),
                "intersection_ours_ratio": intersection_ours_ratio,
                "intersection_baseline_ratio": intersection_baseline_ratio,
                "ours_clips": ours_clip,
                "ours_answer": ours_data["agent_answer"],
                "baseline_clips": baseline_clip,
                "baseline_answer": baseline_data["agent_answer"],
                "ours_session": ours_session,
                "baseline_session": baseline_session,
            }
        
        assert ours_data["question"] == baseline_data["question"]
        
        ours_correct = ours_data["verify_result"].lower().startswith("yes")
        baseline_correct = baseline_data["verify_result"].lower().startswith("yes")
        
        if not baseline_correct and ours_correct:
            turn_correct.append(data_pair)
        elif baseline_correct and not ours_correct:
            turn_incorrect.append(data_pair)
        elif baseline_correct and ours_correct:
            both_correct.append(data_pair)
        else:
            both_incorrect.append(data_pair)
    
    print(f"turn_correct: {len(turn_correct)}, turn_incorrect: {len(turn_incorrect)}, both_correct: {len(both_correct)}, both_incorrect: {len(both_incorrect)}")

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
    parser.add_argument('--answers_only', action='store_true', help='Only print answers')
    args = parser.parse_args()
    
    check_diff(args.ours, args.baseline, args.output_dir, args.answers_only)

if __name__ == "__main__":
    main()
    