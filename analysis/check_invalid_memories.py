import json
import sys
import os
import argparse
from mmagent.utils.general import load_video_graph
from mmagent.retrieve import translate
import mmagent.videograph
sys.modules["videograph"] = mmagent.videograph

def check_invalid_memories(video_graph):
    video_graph.refresh_equivalences()
    invalid_num = 0
    mems = [video_graph.nodes[node_id].metadata['contents'][0] for node_id in video_graph.text_nodes]
    translated_mems = translate(video_graph, mems)
    for translated_mem in translated_mems:
        if "<face_" in translated_mem or "<voice_" in translated_mem:
            invalid_num += 1
    return invalid_num, len(translated_mems)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mems_dir", type=str, default="/mnt/hdfs/foundation/longlin.kylin/mmagent/data/mems/CZ_1")
    args = parser.parse_args()
    
    mems_dir = args.mems_dir
    mems_paths = [os.path.join(mems_dir, f) for f in os.listdir(mems_dir) if f.endswith(".pkl")]
    total_invalid_num = 0
    total_num = 0
    for mems_path in mems_paths:
        video_graph = load_video_graph(mems_path)
        invalid_num, total_num = check_invalid_memories(video_graph)
        total_invalid_num += invalid_num
        total_num += total_num
    print(f"Invalid memories ratio: {total_invalid_num / total_num}")

if __name__ == "__main__":
    main()