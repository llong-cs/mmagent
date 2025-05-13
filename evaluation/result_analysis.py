import json
import os
import argparse

def get_filtered_questions(test_file):
    filtered_questions = []
    with open(test_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if not data['flag']:
                filtered_questions.append(data['question'])
    print(f"Filtered {len(filtered_questions)} questions from {test_file}")
    return filtered_questions

def analyze_agent_results(result_dir):
    agent_results = os.listdir(result_dir)
    agent_results = [os.path.join(result_dir, f) for f in agent_results if "verified" in f]
    agent_result_files = [open(result, "r") for result in agent_results]
    
    print(f"Found {len(agent_results)} agent results in {result_dir}")
    
    for k in range(1, len(agent_results) + 1):
        result_files = agent_result_files[:k]
        
        total = 0
        correct = 0
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
                    flag = qa["flag"]
                    qa_attempts.append(qa["verify_result"].lower().startswith("yes"))

            # If we've reached end of files, break
            if not qa_attempts:
                break

            # Count as correct if any attempt was successful
            # if question not in filtered_questions:
            if not flag:
                total += 1
                if any(qa_attempts):
                    correct += 1
            else:
                filtered_count += 1

            idx += 1

        # Close all files
        for f in result_files:
            f.close()

        print(f"Pass@{k}: {correct / total}")
        print(filtered_count)
        

def parse_args():
    parser = argparse.ArgumentParser(description='Result analysis')
    parser.add_argument('--result', type=str, help='Path to the result directory or file', default="data/annotations/results/5_rounds_threshold_0_top3_no_planning")
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
    
    