import json
import sys
import os
import argparse
from tqdm import tqdm
import re
from mmagent.utils.general import load_video_graph
from mmagent.retrieve import translate
import mmagent.videograph
sys.modules["videograph"] = mmagent.videograph

def check_invalid_memories(video_graph, mode):
    video_graph.refresh_equivalences()
    invalid_face_num = 0
    invalid_voice_num = 0
    num = 0
    if mode == "node":
        mems = [video_graph.nodes[node_id].metadata['contents'][0] for node_id in video_graph.text_nodes]
        translated_mems = translate(video_graph, mems)
        for translated_mem in translated_mems:
            pattern = r'<([^<>]*_[^<>]*)>'
            entity_strs = re.findall(pattern, translated_mem)
            if len(entity_strs) > 0:
                num += 1
                if "<face_" in translated_mem:
                    invalid_face_num += 1
                if "<voice_" in translated_mem:
                    invalid_voice_num += 1
    elif mode == "clip":
        for _, clip_nodes in video_graph.text_nodes_by_clip.items():
            mems = [video_graph.nodes[node_id].metadata['contents'][0] for node_id in clip_nodes]
            translated_mems = translate(video_graph, mems)
            translated_mems = " ".join(translated_mems)
            pattern = r'<([^<>]*_[^<>]*)>'
            entity_strs = re.findall(pattern, translated_mems)
            if len(entity_strs) > 0:
                num += 1
                if "<face_" in translated_mems:
                    invalid_face_num += 1
                if "<voice_" in translated_mems:
                    invalid_voice_num += 1
    else:
        raise ValueError(f"Invalid mode: {mode}")
    # print(invalid_num, len(translated_mems))
    return invalid_face_num, invalid_voice_num, num

def check_unmatched_ids(outputs_dir):
    for file in tqdm(os.listdir(outputs_dir)):
        if file.endswith(".json"):
            with open(os.path.join(outputs_dir, file), "r") as f:
                data = json.load(f)
                input = data[0]
                output = data[1]
                input_str = ""
                for c in input["content"]:
                    if c["type"] == "text":
                        input_str += c["text"]
                        input_str += "\n"
                output_str = output["content"][0]["text"]
                pattern = r'<([^<>]*_[^<>]*)>'
                entity_strs = re.findall(pattern, output_str)
                for entity_str in entity_strs:
                    if entity_str not in input_str:
                        print(file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mems_dir", type=str, default="/mnt/hdfs/foundation/longlin.kylin/mmagent/data/mems_qwen_0511/CZ_1")
    parser.add_argument("--outputs_dir", type=str, default="/mnt/bn/videonasi18n/longlin.kylin/mmagent/data/sft/memgen/0511/evaluation/checkpoint-2214/val_gen")
    parser.add_argument("--mode", type=str, default="node")
    args = parser.parse_args()
    
    mems_dir = args.mems_dir
    mems_paths = [os.path.join(mems_dir, f) for f in os.listdir(mems_dir) if f.endswith(".pkl")]
    total_invalid_face_num = 0
    total_invalid_voice_num = 0
    total_num = 0
    for mems_path in tqdm(mems_paths):
        video_graph = load_video_graph(mems_path)
        invalid_face_num, invalid_voice_num, num = check_invalid_memories(video_graph, args.mode)
        total_invalid_face_num += invalid_face_num
        total_invalid_voice_num += invalid_voice_num
        total_num += num
    print(total_invalid_face_num)
    print(total_invalid_voice_num)
    print(total_num)
    print(f"Invalid face id ratio: {total_invalid_face_num / total_num}")
    print(f"Invalid voice id ratio: {total_invalid_voice_num / total_num}")
    print(f"Invalid id ratio: {(total_invalid_face_num + total_invalid_voice_num) / total_num}")
    
    # check_unmatched_ids(args.outputs_dir)

if __name__ == "__main__":
    main()