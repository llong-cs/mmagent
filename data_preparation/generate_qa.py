from mmagent.utils.general import load_video_graph, validate_and_fix_json
import mmagent.videograph
from mmagent.retrieve import translate
from mmagent.prompts import prompt_generate_qa_from_route
from mmagent.utils.chat_api import generate_messages, get_response_with_retry, parallel_get_response
import json
import os
import argparse
import sys

sys.modules["videograph"] = mmagent.videograph

processing_config = json.load(open("config/processing_config.json", "r"))
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
            "content": prompt_generate_qa_from_route.format(events=route_contents)
        }
    ]
    message = generate_messages(input)
    model = "gpt-4o-2024-11-20"
    qa = None
    for i in range(MAX_RETRIES):
        response = get_response_with_retry(model, message)
        qa = validate_and_fix_json(response)
        if qa is not None:
            break
    return qa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_path", type=str, help="The path to the memory file", default="/mnt/hdfs/foundation/longlin.kylin/mmagent/data/mems_qwen/CZ_1/EMpaw46kr2w_30_5_-1_10_20_0.3_0.6.pkl")
    args = parser.parse_args()

    route, route_contents = get_route_from_mem(args.mem_path, length=10)
    print(route_contents)
    
    qa = generate_qa_from_route(route, route_contents)
    print(qa)
    
if __name__ == "__main__":
    main()