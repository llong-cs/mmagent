from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration, GenerationConfig
from qwen_omni_utils import process_mm_info
import json
import os
import re
import sys
from mmagent.retrieve import back_translate, translate, verify_qa
from mmagent.utils.chat_api import parallel_get_embedding
from mmagent.utils.general import load_video_graph
import mmagent.videograph
sys.modules["videograph"] = mmagent.videograph
import torch
import numpy as np
torch.set_printoptions(threshold=np.inf)

model_path = sys.argv[1]
index = sys.argv[2]
times = sys.argv[3]
os.system(f"mkdir {model_path}/output")
# save_path = model_path.strip("/").split("/")[-1]
prompt = "You are given a question and some relevant knowledge about a specific video. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query to search for relevant knowledge.\n"
pattern = r"\[(.*)\](.*)"
test_path = "/mnt/bn/videonasi18n/longlin.kylin/mmagent/data/annotations/small_test.jsonl"
# model_path = "/mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/sft_output"
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

def eval_answer(question, predict, ground_truth):
    if predict == "":
        return False
    response = verify_qa(question, ground_truth, predict).lower()
    return True if "yes" in response else False

def search(query, video_graph, history_clip=set(), threshold=0.05):
    model = "text-embedding-3-large"
    queries = back_translate(video_graph, [query])
    query_embedding = parallel_get_embedding(model, queries)[0]
    nodes = video_graph.search_text_nodes(query_embedding, threshold=threshold)
    nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    resp = f'Search results of query "{query}":\n\n'
    resp_len = len(resp)
    _clip = set()
    raw_data = list()
    for node in nodes:
        if len(_clip) == 5:
            break
        node_id = node[0]
        node_score = node[1]
        clip_id = video_graph.nodes[node_id].metadata['timestamp']
        if clip_id in _clip:
            continue
        _clip.add(clip_id)
        if clip_id in history_clip:
            continue
        history_clip.add(clip_id)
        clip_node_id = video_graph.text_nodes_by_clip[clip_id]
        clip_node_id = sorted(clip_node_id)
        
        content = translate(video_graph, [video_graph.nodes[_node_id].metadata['contents'][0] for _node_id in clip_node_id])
        text = '\n'.join(content)

        raw_data.append({'clip_id': 'clip_' + str(clip_id), 'memory': content})
        
        resp = resp + 'ID=' + str(clip_id) + '\n' + text + '\n\n'
    if len(resp) < resp_len + 5:
        resp = resp + 'No results found.\n\n'
    return resp, history_clip, raw_data

with open(test_path) as f, open(f"{model_path}/output/{times}_{index}.jsonl", "w") as f1:
    for i, line in enumerate(f.readlines()):
        if i % 8 != int(index):
            continue
        data = json.loads(line)
        video_path = os.path.join(data["clip_path"], str(max([int(i[:-4]) for i in os.listdir(data["clip_path"])]) - 1) + ".mp4")
        video_graph = load_video_graph(data["mem_path"])
        conversation = [[
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
        ]]
        
        history_clip = set()
        for i in range(10):
            add_generation_prompt = True if i == 0 else False
            text = processor.apply_chat_template(conversation, add_generation_prompt=add_generation_prompt, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
            inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
            generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.7,
                top_k=10,
                top_p=0.8,
                pad_token_id=151643,
                bos_token_id=151644,
                eos_token_id=151644
            )
            inputs = inputs.to(model.device).to(model.dtype)
            text_ids = model.generate(**inputs, generation_config=generation_config, use_audio_in_video=True, max_new_tokens=256)
            output = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if len(conversation[0]) == 1:
                conversation[0].append({
                    "role": "assistant",
                    "content": ""
                })
            conversation[0][1]["content"] += "<think>" + output[0].split("<think>")[-1]
            match_result = re.search(pattern, output[0].split("</think>")[-1], re.DOTALL)
            if match_result:
                action = match_result.group(1)
                action_content = match_result.group(2)
            else:
                action = "Search"
                action_content = output[0].split("</think>")[-1]
            if action == "Answer":
                data["model_answer"] = action_content
                data["session"] = conversation[0]
                data["gpt_eval"] = eval_answer(data["question"], action_content, data["answer"])
                break
            else:
                _, history_clip, raw_data = search(action_content, video_graph, history_clip)
                conversation[0][1]["content"] += "<|im_start|>Searched knowledge:\n" + json.dumps(raw_data, ensure_ascii=False) + "\n"
                data["session"] = conversation[0]
        f1.write(json.dumps(data, ensure_ascii=False) + '\n')
        f1.flush()
