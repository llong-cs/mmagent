from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import json
import sys
import re
from vllm import LLM, SamplingParams
from mmagent.retrieve import search, verify_qa
from mmagent.utils.general import load_video_graph
import mmagent.videograph
sys.modules["videograph"] = mmagent.videograph

prompt_generate_action = """You are given a question and some relevant knowledge about a specific video. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank that contains detailed descriptions and high-level abstractions of the video, considering the question, and the provided knowledge.

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If the answer can be derived from the provided knowledge, the {{content}} is the specific answer to the question
If the answer cannot be derived yet, the {{content}} should be a single search query that would help retrieve the missing information.
You need to provide an answer within 5 rounds.

Question: {question}"""

model_name="/mnt/hdfs/foundation/agent/heyc/ckpts/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
import sys
index = int(sys.argv[1])
infer_type = "vllm"
if infer_type == "vllm":
    model = LLM(model=model_name)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name)

def eval_answer(question, predict, ground_truth):
    if predict == "":
        return False
    response = verify_qa(question, ground_truth, predict).lower()
    return True if "yes" in response else False

class QwenChatbot:
    def __init__(self, question, final_round=5):
        self.tokenizer = tokenizer
        self.model = model
        self.final_round = final_round
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            min_p=0,
            pad_token_id=151643,
            bos_token_id=151644,
            eos_token_id=151645
        )
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=4096
        )
        self.history = [{"role": "user", "content": prompt_generate_action.format(question=question)}, {"role": "user", "content": "Searched knowledge: []"}]

    def generate_response(self, rounds):
        self.history[-1]["content"] += f"\n\nRound {rounds}:"
        if rounds == self.final_round:
            self.history[-1]["content"] += " (the Action should be [Answer])"

        text = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True
        )

        if infer_type == "vllm":
            outputs = self.model.generate([text], self.sampling_params)
            response = outputs[0].outputs[0].text
        else:
            inputs = self.tokenizer(text, return_tensors="pt")
            response_ids = self.model.generate(**inputs, generation_config=self.generation_config, max_new_tokens=512)[0][len(inputs.input_ids[0]):].tolist()
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # outputs = self.model.generate([text], self.sampling_params)
        # print(outputs.outputs[0].text)

        # Update history
        self.history.append({"role": "assistant", "content": response})
        # return response

    def add_search_result(self, search_result):
        self.history.append({"role": "user", "content": search_result})

total_round = 10
# Example Usage
if __name__ == "__main__":
    test_path = "/mnt/bn/videonasi18n/longlin.kylin/mmagent/data/annotations/small_test.jsonl"
    with open(test_path) as f, open(f"/mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/output/qwen3_10/{index}.jsonl", "w") as f1:
        for idx, line in enumerate(f.readlines()):
            if idx % 16 != index:
                continue
            data = json.loads(line)
            video_graph = load_video_graph(data["mem_path"])
            question = data["question"]
            data["memory_scores"] = []
            # print(data["question"])
            chatbot = QwenChatbot(question, total_round)
            currenr_clips = []

            for i in range(total_round):
                chatbot.generate_response(i + 1)
                pattern = r"Action: \[(.*)\].*Content: (.*)"
                output = chatbot.history[-1]["content"].split("</think>")[-1]
                match_result = re.search(pattern, output, re.DOTALL)
                if match_result:
                    action = match_result.group(1)
                    content = match_result.group(2)
                else:
                    action = "Search"
                    content = output
                if action == "Answer":
                    data["model_answer"] = content
                    data["gpt_eval"] = eval_answer(data["question"], data["model_answer"], data["answer"])
                    break
                else:
                    new_memories, currenr_clips, clip_scores = search(video_graph, content, currenr_clips)
                    chatbot.add_search_result("Searched knowledge: " + json.dumps(new_memories, ensure_ascii=False))
                    data["memory_scores"].append(clip_scores)
                data["session"] = chatbot.history
            f1.write(json.dumps(data, ensure_ascii=False) + "\n")