import json
import os
import argparse

def analyze_agent_results(result_dir):
    agent_results = os.listdir(result_dir)
    agent_results = [os.path.join(result_dir, f) for f in agent_results if "verified" in f]
    agent_result_files = [open(result, "r") for result in agent_results]
    
    print(f"Found {len(agent_results)} agent results in {result_dir}")
    
    for k in range(1, len(agent_results) + 1):
        result_files = agent_result_files[:k]
        
        total = 0
        correct = 0
        total_with_flag = 0
        correct_with_flag = 0
        filtered_count = 0
        idx = 0

        # Read all QA pairs line by line across k files
        while True:
            qa_attempts = []
            # Try to read next line from each file
            for f in result_files:
                line = f.readline()
                if line:
                    qa = json.loads(line)
                    flag = False if "flag" not in qa else qa["flag"]
                    qa_attempts.append(qa["verify_result"].lower().startswith("yes"))

            # If we've reached end of files, break
            if not qa_attempts:
                break

            # Count as correct if any attempt was successful
            total += 1
            if any(qa_attempts):
                correct += 1
            
            if not flag:
                total_with_flag += 1
                if any(qa_attempts):
                    correct_with_flag += 1
            else:
                filtered_count += 1

            idx += 1

        # Close all files
        for f in result_files:
            f.close()

        print(f"Pass@{k} (hard-{total_with_flag}): {correct_with_flag / total_with_flag:.4f}")
        print(f"Pass@{k} (all-{total}): {correct / total:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description='Result analysis')
    parser.add_argument('--result', type=str, help='Path to the result directory or file', default="data/annotations/results/gemini-1.5-pro-002/5_rounds_threshold_0.5_top5_no_planning_qwen_0511")
    parser.add_argument('--model', type=str, help='Model name', default="agent")
    parser.add_argument('--test_file', type=str, help='Path to the test file', default="data/annotations/small_test.jsonl")
    
    return parser.parse_args()

def main():
    args = parse_args()
    result = args.result
    model = args.model
    
    if model == "agent":
        analyze_agent_results(result)
    else:
        pass
    
if __name__ == "__main__":
    main()

# an example to run the code
# python3 evaluation/result_analysis.py --result data/annotations/results/gpt-4o-2024-11-20/5_rounds_threshold_0.5_top2_no_planning_qwen_0511 --test_file data/annotations/small_test.jsonl
    