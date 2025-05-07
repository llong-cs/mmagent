from mmagent.utils.general import load_video_graph, validate_and_fix_json
import mmagent.videograph
from mmagent.retrieve import translate
from mmagent.prompts import prompt_generate_qa_from_route
from mmagent.utils.chat_api import generate_messages, get_response_with_retry, parallel_get_response
import json
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
sys.modules["videograph"] = mmagent.videograph

processing_config = json.load(open("configs/processing_config.json", "r"))
MAX_RETRIES = processing_config["max_retries"]

def get_route_from_mem(mem_path, length=10):
    mem = load_video_graph(mem_path)
    route, route_contents = mem.sample_a_route(length=length)
    route_contents = translate(mem, route_contents)
    return route, route_contents

def generate_qa_from_route(route, route_contents):
    input = [
        {
            "type": "text",
            "content": prompt_generate_qa_from_route.format(events=[
                {
                    "id": node_id,
                    "content": content
                }
                for node_id, content in zip(route, route_contents)
            ])
        }
    ]
    message = generate_messages(input)
    model = "gpt-4o-2024-11-20"
    res = None
    for i in range(MAX_RETRIES):
        response = get_response_with_retry(model, message)[0]
        res = validate_and_fix_json(response)
        if res is not None:
            break
    return res

def generate_qa_from_mem(mem_path, length=10):
    route, route_contents = get_route_from_mem(mem_path, length)
    res = generate_qa_from_route(route, route_contents)
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_path", type=str, help="The path to the memory file", default="/mnt/hdfs/foundation/longlin.kylin/mmagent/data/mems_qwen/CZ_1/EMpaw46kr2w_30_5_-1_10_20_0.3_0.6.pkl")
    parser.add_argument("--length", type=int, help="The length of the route", default=10)
    args = parser.parse_args()
    
    length = args.length

    mem_paths = []
    with open("data/annotations/train_500.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            if data["mem_path"] not in mem_paths:
                mem_paths.append(data["mem_path"])
    mem_paths = mem_paths + mem_paths

    qas_from_mems = []
    
    def process_mem(mem_path):
        res = generate_qa_from_mem(mem_path, length)
        return {
            "mem_path": mem_path,
            "question": res["question"],
            "answer": res["answer"], 
            "related_ids": res["related_ids"]
        }

    with ProcessPoolExecutor(max_workers=8) as executor:
        qas_from_mems = list(tqdm(
            executor.map(process_mem, mem_paths),
            total=len(mem_paths),
            desc="Generating QAs from memories"
        ))
    
    with open(f"data/sft/actgen/qas_from_mems/qas_length_{length}.jsonl", "w") as f:
        for qa in qas_from_mems:
            f.write(json.dumps(qa) + "\n")
    # lengths = [5, 10, 15, 20, 25, 30]

    # for length in lengths:
    #     print(f"length: {length}")
    #     route, route_contents, res = generate_qa_from_mem(args.mem_path, length)
    #     print(res)
    #     print(f"related nodes: {route}")
    #     print("-"*20)
    
if __name__ == "__main__":
    main()