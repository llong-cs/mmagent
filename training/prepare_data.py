import os
import json

prompt = "You are given a question and some relevant knowledge about a specific video. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query to search for relevant knowledge.\n"

data_path = "/mnt/bn/videonasi18n/longlin.kylin/mmagent/data/annotations/train_500.jsonl"
output_path = "data/ppo_data.jsonl"

with open(data_path) as f, open(output_path, "w") as f1:
    for line in f.readlines():
        data = json.loads(line)
        video_path = os.path.join(data["clip_path"], str(max([int(i[:-4]) for i in os.listdir(data["clip_path"])]) - 1) + ".mp4")

        res = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "video",
                            "video": video_path
                        },
                        {
                            "type": "text",
                            "text": "Question: " + data["question"]
                        }
                    ]
                }
            ]
        }
        
        out_str = ""
        if "session" in data:
            for i, (knowledge, output) in enumerate(zip(*data["session"])):
                if i > 0:
                    out_str += "Searched knowledge:\n" + json.dumps(knowledge, ensure_ascii=False) + "\n"
                if output["action_type"] == "answer":
                    out_str += "<think>" + output["reasoning"] + "</think>\n[" + output["action_type"].capitalize() + "]" + data["answer"] + "<|im_start|>"
                else:
                    out_str += "<think>" + output["reasoning"] + "</think>\n[" + output["action_type"].capitalize() + "]" + output["action_content"] + "<|im_start|>"
            res["messages"].append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": out_str
                    }
                ]
            })
        else:
            res["answer"] = data["answer"]
            res["mem"] = data["mem_path"]
        
        f1.write(json.dumps(res, ensure_ascii=False) + "\n")

# import torch
# import numpy as np
# torch.set_printoptions(threshold=np.inf)
# from qwen_omni_utils import process_mm_info
# from transformers import Qwen2_5OmniProcessor
# processor = Qwen2_5OmniProcessor.from_pretrained("/mnt/hdfs/foundation/agent/heyc/ckpts/Qwen2.5-Omni-7B-thinker")
# with open("data/sft_data.jsonl") as f:
#     for line in f.readlines():
#         data = json.loads(line)["messages"]
#         text = processor.apply_chat_template(data, tokenize=False)
#         audios, images, videos = process_mm_info(data, use_audio_in_video=True)
#         inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
#         print(inputs["input_ids"])
#         break