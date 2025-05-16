import json
import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

if torch.cuda.is_available():
    processor = Qwen2_5OmniProcessor.from_pretrained("/mnt/hdfs/foundation/agent/heyc/ckpts/Qwen2.5-Omni-7B-thinker")

def count_tokens(file_path):
    vids = []
    count = 0
    total_len = 0
    with open(file_path, "r") as f:
        for line in f:
            total_mem = ""
            data = json.loads(line)
            # if data["video_url"] in vids:
            #     continue
            vids.append(data["video_url"])
            try:
                for mem in data["session"][0]:
                    for s in mem:
                        contents = s["memory"]
                        for e in contents:
                            total_mem += e
                            total_mem += "\n"
                text = processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": total_mem}]}], add_generation_prompt=False, tokenize=True)[0]
                total_len += len(text)
                count += 1
            except:
                continue
    
    
    print(total_len / count)

if __name__ == "__main__":
    count_tokens("data/annotations/results/5_rounds_threshold_0_top3_no_planning/small_test_with_agent_answer_0.jsonl")
