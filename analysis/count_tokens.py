import json
import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

if torch.cuda.is_available():
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        "/mnt/hdfs/foundation/agent/heyc/ckpts/Qwen2.5-Omni-7B-thinker",
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    thinker.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained("/mnt/hdfs/foundation/agent/heyc/ckpts/Qwen2.5-Omni-7B-thinker")

def count_tokens(file_path):
    total_mem = ""
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            for mem in data["session"][0]:
                contents = mem["memory"]
                for e in contents:
                    total_mem += e
                    total_mem += "\n"
    
    text = processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": total_mem}]}], add_generation_prompt=False, tokenize=True)
    print(len(text))

if __name__ == "__main__":
    count_tokens("data/annotations/results/full_retrieval_threshold_0_qwen_0511/small_test_qwen_with_agent_answer_0.jsonl")
