import json
import os
import base64
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor
from utils.general import load_video_graph
from utils.chat_api import generate_messages, get_response_with_retry, parallel_get_response
from retrieve import answer_with_retrieval
from prompts import prompt_agent_verify_answer

def video_to_base64(video_path):
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
        base64_encoded = base64.b64encode(video_bytes).decode('utf-8')
        return base64_encoded

def process_qa(qa, planning=True):
    try:
        mem = load_video_graph(qa["mem_path"])
        question = qa["question"]
        
        if planning:
            clip_path = qa["clip_path"]
            clips = os.listdir(clip_path)
            # sorted by number
            last_clip = sorted(clips, key=lambda x: int(x.split(".")[0]))[-1]
            video_clip_base64 = video_to_base64(os.path.join(clip_path, last_clip))
        else:
            video_clip_base64 = None
        
        agent_answer, session = answer_with_retrieval(mem, question, video_clip_base64)
        qa["agent_answer"] = agent_answer
        qa["session"] = session
    except Exception as e:
        print(f"Error processing qa: {qa['question']}")
        print(e)
        qa["agent_answer"] = None
        qa["session"] = None
        return None
    return qa


def verify_qa(qa):
    try:
        questions = qa["question"]
        ground_truth = qa["answer"]
        agent_answer = qa["agent_answer"]
        qa_sample = {
            "question": questions,
            "ground_truth_answer": ground_truth,
            "agent_answer": agent_answer,
        }

        input = [
            {
                "type": "text",
                "content": json.dumps(qa_sample),
            },
            {
                "type": "text",
                "content": prompt_agent_verify_answer,
            },
            {
                "type": "text",
                "content": "Now answer if the answer from the baseline is correct or not:",
            },
        ]
        messages = generate_messages(input)
        model = "gpt-4o-2024-11-20"
        response = get_response_with_retry(model, messages)
        qa["verify_result"] = response[0]
    except Exception as e:
        print(f"Error verifying qa: {qa['question']}")
        print(e)
        qa["verify_result"] = None
        return None
    return qa

def process_qa_list(qa_list, dataset_with_agent_answer, max_workers=16):
    bs = 100
    results = []
    try:
        with open(dataset_with_agent_answer, "r") as f:
            sample_count = len(f.readlines())
    except Exception as e:
        print(f"Error reading dataset_with_agent_answer: {dataset_with_agent_answer}")
        print(e)
        sample_count = 0
    for i in range(sample_count, len(qa_list), bs):
        qa_list_batch = qa_list[i:i+bs]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            result = list(
                tqdm(
                    executor.map(process_qa, qa_list_batch),
                    total=len(qa_list_batch),
                    desc=f"Generating answers {i}/{len(qa_list)}",
                )
            )
        with open(dataset_with_agent_answer, "a") as f:
            for qa in result:
                f.write(json.dumps(qa) + "\n")
        results.extend(result)
    return results

def verify_qa_list(qa_list, dataset_with_agent_answer_verified):
    bs = 100
    try:
        with open(dataset_with_agent_answer_verified, "r") as f:
            sample_count = len(f.readlines())
    except Exception as e:
        print(f"Error reading dataset_with_agent_answer_verified: {dataset_with_agent_answer_verified}")
        print(e)
        sample_count = 0
    for i in range(sample_count, len(qa_list), bs):
        qa_list_batch = qa_list[i:i+bs]
        inputs = [
            [
                {
                    "type": "text",
                    "content": json.dumps({
                        "question": qa["question"],
                        "ground_truth_answer": qa["answer"],
                        "agent_answer": qa["agent_answer"],
                    }),
                },
                {
                    "type": "text",
                    "content": prompt_agent_verify_answer,
                },
                {
                    "type": "text",
                    "content": "Now answer if the answer from the baseline is correct or not:",
                },            
            ] for qa in qa_list_batch
        ]
        messages = [generate_messages(input) for input in inputs]
        model = "gpt-4o-2024-11-20"
        responses = parallel_get_response(model, messages)

        verify_results = responses[0]
        for qa, verify_result in zip(qa_list_batch, verify_results):
            qa["verify_result"] = verify_result
        
        with open(dataset_with_agent_answer_verified, "a") as f:
            for qa in qa_list_batch:
                f.write(json.dumps(qa) + "\n")
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/annotations/small_train.jsonl")
    parser.add_argument("--sample_rounds", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="data/annotations/results")
    
    
    args = parser.parse_args()
    args.dataset_with_agent_answer = os.path.join(args.output_dir, os.path.basename(args.dataset).replace(".jsonl", "_with_agent_answer.jsonl"))
    args.dataset_with_agent_answer_verified = os.path.join(args.output_dir, os.path.basename(args.dataset_with_agent_answer).replace("_with_agent_answer", "_with_agent_answer_verified"))

    qa_list = []
    dataset = args.dataset
    with open(dataset, "r") as f:
        for line in f:
            qa = json.loads(line)
            if os.path.exists(qa["mem_path"]):
                qa_list.append(qa)

    # # idx = 0
    # # qa_list_with_agent_answer = process_qa_list(qa_list[idx:idx+1])
    sample_rounds = args.sample_rounds
    for i in range(sample_rounds):
        dataset_with_agent_answer = args.dataset_with_agent_answer.replace("_with_agent_answer", f"_with_agent_answer_{i}")
        dataset_with_agent_answer_verified = args.dataset_with_agent_answer_verified.replace("_with_agent_answer_verified", f"_with_agent_answer_verified_{i}")
        # clear the file
        # with open(dataset_with_agent_answer, "w") as f:
        #     f.truncate(0)
        qa_list_with_agent_answer = process_qa_list(qa_list, dataset_with_agent_answer)
        # clear the file
        # with open(dataset_with_agent_answer_verified, "w") as f:
        #     f.truncate(0)
        verify_qa_list(qa_list_with_agent_answer, dataset_with_agent_answer_verified)
