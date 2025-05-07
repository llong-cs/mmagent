from mmagent.utils.general import load_video_graph
import mmagent.videograph
from mmagent.retrieve import translate
import json
import os
import argparse
import sys

sys.modules["videograph"] = mmagent.videograph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_path", type=str, help="The path to the memory file", default="/mnt/hdfs/foundation/longlin.kylin/mmagent/data/mems_qwen/CZ_1/EMpaw46kr2w_30_5_-1_10_20_0.3_0.6.pkl")
    args = parser.parse_args()

    mem = load_video_graph(args.mem_path)
    route, route_contents = mem.sample_a_route(length=10)
    route_contents = translate(mem, route_contents)
    
    print(route_contents)
    
if __name__ == "__main__":
    main()