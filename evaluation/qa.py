import json
import os
import base64
from tqdm import tqdm
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
import logging

from mmagent.utils.general import load_video_graph
from mmagent.utils.chat_api import generate_messages, get_response_with_retry, parallel_get_response
from mmagent.retrieve import answer_with_retrieval
from mmagent.prompts import prompt_agent_verify_answer, prompt_agent_verify_answer_with_reasoning, prompt_agent_verify_answer_referencing
import mmagent.videograph

# Configure logger
logger = logging.getLogger(__name__)

sys.modules["videograph"] = mmagent.videograph

processing_config = json.load(open("configs/processing_config.json"))

def video_to_base64(video_path):
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
        base64_encoded = base64.b64encode(video_bytes).decode('utf-8')
        return base64_encoded

def process_qa(qa):
    try:
        mem = load_video_graph(qa["mem_path"])

        # refresh equivalences
        mem.refresh_equivalences()

        question = qa["question"]
        planning = processing_config["planning"]
        
        if planning:
            clip_path = qa["clip_path"]
            clips = os.listdir(clip_path)
            # sorted by number
            last_clip = sorted(clips, key=lambda x: int(x.split(".")[0]))[-2]
            video_clip_base64 = video_to_base64(os.path.join(clip_path, last_clip))
        else:
            video_clip_base64 = None
        
        before_clip = qa.get("before_clip", None)
        
        agent_answer, session = answer_with_retrieval(
            mem, 
            question, 
            video_clip_base64, 
            topk=processing_config["topk"], 
            multiple_queries=processing_config["multiple_queries"], 
            max_retrieval_steps=processing_config["max_retrieval_steps"], 
            route_switch=processing_config["route_switch"],
            threshold=processing_config["retrieval_threshold"],
            model=args.retrieval_model,
            before_clip=before_clip,
        )
        
        qa["agent_answer"] = agent_answer
        qa["session"] = session
    except Exception as e:
        logger.error(f"Error processing sample: {json.dumps(qa)}")
        logger.error(str(e))
        qa["agent_answer"] = None
        qa["session"] = None
        return qa
    return qa

def process_qa_list(qa_list, dataset_with_agent_answer, max_workers=32):
    bs = 100
    results = []
    try:
        with open(dataset_with_agent_answer, "r") as f:
            sample_count = len(f.readlines())
    except Exception as e:
        logger.error(f"Error reading dataset_with_agent_answer: {dataset_with_agent_answer}")
        logger.error(str(e))
        sample_count = 0
    logger.info(f"Starting from sample {sample_count}")
    logger.info(f"Processing {len(qa_list)} samples with batch size {bs}")
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
        logger.error(f"Error reading dataset_with_agent_answer_verified: {dataset_with_agent_answer_verified}")
        logger.error(str(e))
        sample_count = 0
    for i in tqdm(range(sample_count, len(qa_list), bs)):
        try:
            qa_list_batch = qa_list[i:i+bs]
            inputs = [
                [
                    {
                        "type": "text",
                        "content": prompt_agent_verify_answer_referencing.format(
                            question=qa["question"],
                            ground_truth_answer=qa["answer"],
                            agent_answer=qa["agent_answer"],
                        ),
                    }          
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
        logger.error(f"Error reading dataset_with_agent_answer_verified: {dataset_with_agent_answer_verified}")
        logger.error(str(e))
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
    parser.add_argument("--sample_rounds", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="data/annotations/results")
    parser.add_argument("--version", type=str, default="0511")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--retrieval_model", type=str, default="gpt-4o-2024-11-20")

    args = parser.parse_args()
    args.dataset_with_agent_answer = os.path.basename(args.dataset).replace(".jsonl", "_with_agent_answer.jsonl")
    args.dataset_with_agent_answer_verified = os.path.basename(args.dataset_with_agent_answer).replace("_with_agent_answer", "_with_agent_answer_verified")
    
    def get_exp_name():
        return f"{processing_config['max_retrieval_steps']}_rounds_threshold_{processing_config['retrieval_threshold']}_top{processing_config['topk']}_{'no_planning' if not processing_config['planning'] else 'planning'}_{args.model}{'_'+args.version if args.model=='qwen' else ''}"

    exp_settings = [
        {
            "max_retrieval_steps": 5,
            "retrieval_threshold": 0.5,
            "planning": False,
            "topk": 2
        },
        {
            "max_retrieval_steps": 5,
            "retrieval_threshold": 0.5,
            "planning": False,
            "topk": 5
        },
    ]

    sample_rounds = args.sample_rounds
    for exp_setting in exp_settings:
        for param, value in exp_setting.items():
            processing_config[param] = value
        for i in range(sample_rounds):
            qa_list = []
            dataset = args.dataset
            with open(dataset, "r") as f:
                for line in f:
                    qa = json.loads(line)
                    if args.model == "qwen":
                        qa["mem_path"] = qa["mem_path"].replace("mems", f"mems_qwen_{args.version}")
                    if os.path.exists(qa["mem_path"]):
                        qa_list.append(qa)
            logger.info(f"Processing {len(qa_list)} samples")

            dataset_with_agent_answer = args.dataset_with_agent_answer.replace("_with_agent_answer", f"_with_agent_answer_{i}")
            dataset_with_agent_answer = os.path.join(args.output_dir, args.retrieval_model, get_exp_name(), dataset_with_agent_answer)
            os.makedirs(os.path.dirname(dataset_with_agent_answer), exist_ok=True)

            dataset_with_agent_answer_verified = args.dataset_with_agent_answer_verified.replace("_with_agent_answer_verified", f"_with_agent_answer_verified_{i}")
            dataset_with_agent_answer_verified = os.path.join(args.output_dir, args.retrieval_model, get_exp_name(), dataset_with_agent_answer_verified)
            os.makedirs(os.path.dirname(dataset_with_agent_answer_verified), exist_ok=True)

            # clear the file
            # with open(dataset_with_agent_answer, "w") as f:
            #     f.truncate(0)
            qa_list = process_qa_list(qa_list, dataset_with_agent_answer)
            qa_list = []
            with open(dataset_with_agent_answer, "r") as f:
                for line in f:
                    try:
                        qa_list.append(json.loads(line))
                    except Exception as e:
                        logger.error(f"Error loading qa: {line}")
                        raise e

            # clear the file
            # with open(dataset_with_agent_answer_verified, "w") as f:
            #     f.truncate(0)
            
            verify_qa_list(qa_list, dataset_with_agent_answer_verified)
            # verify_qa_list_with_reasoning(qa_list_with_agent_answer, dataset_with_agent_answer_verified)