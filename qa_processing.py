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
    return qa


def verify_qa(qa):
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

    return qa

def process_qa_list(qa_list, max_workers=16):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        qa_list = list(
            tqdm(
                executor.map(process_qa, qa_list),
                total=len(qa_list),
                desc="Generating answers",
            )
        )
    return qa_list

def verify_qa_list(qa_list):
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
        ] for qa in qa_list
    ]
    messages = [generate_messages(input) for input in inputs]
    model = "gpt-4o-2024-11-20"
    responses = parallel_get_response(model, messages)

    results = responses[0]
    for qa, result in zip(qa_list, results):
        qa["verify_result"] = result

    return qa_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/annotations/small_test.jsonl")
    args = parser.parse_args()
    args.dataset_with_agent_answer = args.dataset.replace(".jsonl", "_with_agent_answer.jsonl")
    args.dataset_with_agent_answer_verified = args.dataset_with_agent_answer.replace("_with_agent_answer", "_with_agent_answer_verified")
    dataset = args.dataset
    dataset_with_agent_answer = args.dataset_with_agent_answer
    dataset_with_agent_answer_verified = args.dataset_with_agent_answer_verified

    qa_list = []
    
    with open(dataset, "r") as f:
        for line in f:
            qa = json.loads(line)
            if os.path.exists(qa["mem_path"]):
                qa_list.append(qa)

    # # idx = 0
    # # qa_list_with_agent_answer = process_qa_list(qa_list[idx:idx+1])
    for i in range(5):
        dataset_with_agent_answer = dataset.replace(".jsonl", f"_with_agent_answer_{i}.jsonl")
        dataset_with_agent_answer_verified = dataset_with_agent_answer.replace("_with_agent_answer", f"_with_agent_answer_verified")
        qa_list_with_agent_answer = process_qa_list(qa_list)

        with open(dataset_with_agent_answer, "w") as f:
            for qa in qa_list_with_agent_answer:
                f.write(json.dumps(qa) + "\n")
            
        qa_list_with_agent_answer_verified = verify_qa_list(qa_list_with_agent_answer)

        with open(dataset_with_agent_answer_verified, "w") as f:
            for qa in qa_list_with_agent_answer_verified:
                f.write(json.dumps(qa) + "\n")
    
    # verify only
    # qa_list = []
    # with open(dataset_with_agent_answer, "r") as f:
    #     for line in f:
    #         qa = json.loads(line)
    #         if os.path.exists(qa["mem_path"]):
    #             qa_list.append(qa)
    
    # qa_list = verify_qa_list(qa_list)
    
    # with open(dataset_with_agent_answer_verified, "w") as f:
    #     for qa in qa_list:
    #         f.write(json.dumps(qa) + "\n")

    # calculate accuracy
    # total = 0
    # correct = 0
    # with open(dataset_with_agent_answer_verified, "r") as f:
    #     for line in f:
    #         qa = json.loads(line)
    #         total += 1
    #         if qa["verify_result"].lower().startswith("yes"):
    #             correct += 1

    # print(f"Accuracy: {correct / total}")
