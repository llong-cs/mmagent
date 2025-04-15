import json
import os
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor
from utils.general import load_video_graph
from utils.chat_api import generate_messages, get_response_with_retry
from retrieve import answer_with_retrieval
from prompts import prompt_agent_verify_answer


def process_qa(qa):
    mem = load_video_graph(qa["mem_path"])
    question = qa["question"]
    agent_answer, session = answer_with_retrieval(mem, question)
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

def verify_qa_list(qa_list, max_workers=50):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        qa_list = list(
            tqdm(
                executor.map(verify_qa, qa_list),
                total=len(qa_list),
                desc="Verifying answers",
            )
        )
    return qa_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/annotations/small_test.jsonl")
    parser.add_argument("--num_processes", type=int, default=16)
    args = parser.parse_args()
    args.dataset_with_agent_answer = args.dataset.replace(".jsonl", "_with_agent_answer.jsonl")

    qa_list = []
    dataset = args.dataset
    dataset_with_agent_answer = args.dataset_with_agent_answer
    num_processes = args.num_processes

    with open(dataset, "r") as f:
        for line in f:
            qa = json.loads(line)
            if os.path.exists(qa["mem_path"]):
                qa_list.append(qa)

    qa_list_with_agent_answer = process_qa_list(qa_list)

    with open(dataset_with_agent_answer, "w") as f:
        for qa in qa_list_with_agent_answer:
            f.write(json.dumps(qa) + "\n")

    qa_list_with_agent_answer_verified = verify_qa_list(qa_list_with_agent_answer)

    with open(dataset_with_agent_answer, "w") as f:
        for qa in qa_list_with_agent_answer_verified:
            f.write(json.dumps(qa) + "\n")