import json
import os
import base64
from tqdm import tqdm
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

from mmagent.utils.general import load_video_graph
from mmagent.utils.chat_api import generate_messages, get_response_with_retry, parallel_get_response
from mmagent.retrieve import answer_with_retrieval
from mmagent.prompts import prompt_agent_verify_answer, prompt_agent_verify_answer_with_reasoning, prompt_agent_verify_answer_referencing
import mmagent.videograph

sys.modules["videograph"] = mmagent.videograph

processing_config = json.load(open("configs/processing_config.json"))

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
            last_clip = sorted(clips, key=lambda x: int(x.split(".")[0]))[-2]
            video_clip_base64 = video_to_base64(os.path.join(clip_path, last_clip))
        else:
            video_clip_base64 = None
        
        agent_answer, session = answer_with_retrieval(mem, question, video_clip_base64, topk=processing_config["topk"], multiple_queries=processing_config["multiple_queries"], max_retrieval_steps=processing_config["max_retrieval_steps"])
        qa["agent_answer"] = agent_answer
        qa["session"] = session
    except Exception as e:
        print(f"Error processing qa: {qa['question']}")
        print(e)
        qa["agent_answer"] = None
        qa["session"] = None
        return qa
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
        try:
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
        except Exception as e:
            raise RuntimeError(f"Error processing qa_list_batch: {i}") from e
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
    for i in tqdm(range(sample_count, len(qa_list), bs)):
        try:
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
                        "content": prompt_agent_verify_answer_referencing,
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
        except Exception as e:
            raise RuntimeError(f"Error processing qa_list_batch: {i}") from e
                
def verify_qa_list_with_reasoning(qa_list, dataset_with_agent_answer_verified):
    bs = 100
    try:
        with open(dataset_with_agent_answer_verified, "r") as f:
            sample_count = len(f.readlines())
    except Exception as e:
        print(f"Error reading dataset_with_agent_answer_verified: {dataset_with_agent_answer_verified}")
        print(e)
        sample_count = 0
    for i in tqdm(range(sample_count, len(qa_list), bs)):
        try:
            qa_list_batch = qa_list[i:i+bs]
            inputs = [
                [
                    {
                        "type": "text",
                        "content": json.dumps({
                            "question": qa["question"],
                            "ground_truth_answer": qa["answer"],
                            "agent_answer": qa["agent_answer"],
                            "reasoning": qa["reasoning"],
                        }),
                    },
                    {
                        "type": "text",
                        "content": prompt_agent_verify_answer_with_reasoning,
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
        except Exception as e:
            raise RuntimeError(f"Error processing qa_list_batch: {i}") from e
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/annotations/small_test.jsonl")
    parser.add_argument("--sample_rounds", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="data/annotations/results")
    
    exp_settings = {
        "full_retrieval": {
            "topk": 1000,
            "multiple_queries": True,
            "max_retrieval_steps": 2
        },
        "large_retrieval": {
            "topk": 30,
            "multiple_queries": True,
            "max_retrieval_steps": 3
        },
        "multiple_queries": {
            "topk": 5,
            "multiple_queries": True,
            "max_retrieval_steps": 20
        }
    }

    args = parser.parse_args()
    args.dataset_with_agent_answer = os.path.basename(args.dataset).replace(".jsonl", "_with_agent_answer.jsonl")
    args.dataset_with_agent_answer_verified = os.path.basename(args.dataset_with_agent_answer).replace("_with_agent_answer", "_with_agent_answer_verified")

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
    for exp, exp_setting in exp_settings.items():
        for param, value in exp_setting.items():
            processing_config[param] = value
        print(f"Processing {exp} with {exp_setting}")
        for i in range(sample_rounds):
            dataset_with_agent_answer = args.dataset_with_agent_answer.replace("_with_agent_answer", f"_with_agent_answer_{i}")
            dataset_with_agent_answer = os.path.join(args.output_dir, exp, dataset_with_agent_answer)
            os.makedirs(os.path.dirname(dataset_with_agent_answer), exist_ok=True)

            dataset_with_agent_answer_verified = args.dataset_with_agent_answer_verified.replace("_with_agent_answer_verified", f"_with_agent_answer_verified_{i}")
            dataset_with_agent_answer_verified = os.path.join(args.output_dir, exp, dataset_with_agent_answer_verified)
            os.makedirs(os.path.dirname(dataset_with_agent_answer_verified), exist_ok=True)

            # clear the file
            # with open(dataset_with_agent_answer, "w") as f:
            #     f.truncate(0)
            qa_list = process_qa_list(qa_list, dataset_with_agent_answer)

            # clear the file
            # with open(dataset_with_agent_answer_verified, "w") as f:
            #     f.truncate(0)
            # qa_list = []
            # with open(dataset_with_agent_answer, "r") as f:
            #     for line in f:
            #         try:
            #             qa_list.append(json.loads(line))
            #         except Exception as e:
            #             print(f"Error loading qa: {line}")
            #             raise e
            verify_qa_list(qa_list, dataset_with_agent_answer_verified)
            # verify_qa_list_with_reasoning(qa_list_with_agent_answer, dataset_with_agent_answer_verified)