# from utils.chat_api import get_response_with_retry, generate_messages
import json
# from mmagent.prompts import prompt_agent_verify_answer_referencing
import sys
# step = int(sys.argv[1])

# t, c = 0, 0
# t1, c1 = 0, 0
# with open(f"data/agent_out/checkpoint-{step}_noanswer.jsonl", "w") as f1:
#     for i in range(8):
#         with open(f"/mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/data/agent_out/checkpoint-{step}_{i}.jsonl") as f:
# #             for line in f.readlines():
# #                 data = json.loads(line)
# #                 if "model_anwert" not in data:
# #                     f1.write(line)
#             for line in f.readlines():
#                 t += 1
#                 if not data["flag"]:
#                     t1 += 1
#                 data = json.loads(line)
#                 if "model_anwert" not in data:
#                     continue
#                 input = [
#                     {
#                         "type": "text",
#                         "content": json.dumps({
#                             "question": data["question"],
#                             "ground_truth_answer": data["answer"],
#                             "agent_answer": data["model_anwert"],
#                         }),
#                     },
#                     {
#                         "type": "text",
#                         "content": prompt_agent_verify_answer_referencing,
#                     },
#                     {
#                         "type": "text",
#                         "content": "Now answer if the answer from the baseline is correct or not:",
#                     },            
#                 ]

#                 messages = generate_messages(input)
#                 model = "gpt-4o-2024-11-20"
#                 response = get_response_with_retry(model, messages)
#                 result = response[0]
#                 data["gpt_result"] = result
#                 f1.write(json.dumps(data, ensure_ascii=False) + "\n")
#                 if "yes" in result.lower():
#                     if not data["flag"]:
#                         c1 += 1
#                     c += 1
# print(c, t, c / t)
# print(c1, t1, c1 / t1)

# all_data = {}
# with open("/mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/data/agent_out/checkpoint-230_new.jsonl") as f:
#     for line in f.readlines():
#         data = json.loads(line)
#         if "yes" in data["gpt_result"].lower():
#             data["gpt_eval"] = True
#         else:
#             data["gpt_eval"] = False
#         all_data["question"] = json.dumps(data)
# with open("/mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/data/agent_output/checkpoint-230/2.jsonl", "w") as f1, open("/mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/data/agent_output/checkpoint-230/3.jsonl") as f:
#     for line in f.readlines():
#         data = json.loads(line)
#         if data["question"] in all_data:
#             f1.write(all_data[data["question"]])
#         else:
#             f1.write(line)
# exit()
files_num = 1
datas = []
datas_hard = []
for i in range(files_num):
    data, data1 = {}, {}
    with open("/mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/output/qwen3_10.jsonl") as f:
        for line in f.readlines():
            line = json.loads(line)
            data[line["question"]] = line
            if not line["flag"]:
                data1[line["question"]] = line
    datas.append(data)
    datas_hard.append(data1)

for i in range(files_num):
    a, c = 0, 0
    for _, v in datas[i].items():
        if "gpt_eval" in v and v["gpt_eval"]:
            c += 1
        a += 1
    print(i, c, a, c / a)
a, c = 0, 0
for k in datas[0].keys():
    correct = False
    for i in range(files_num):
        if "gpt_eval" in datas[i][k] and datas[i][k]["gpt_eval"]:
            correct = True
    if correct:
        c += 1
    a += 1
print(c, a, c / a)
    
for i in range(files_num):
    a, c = 0, 0
    for _, v in datas_hard[i].items():
        if "gpt_eval" in v and v["gpt_eval"]:
            c += 1
        a += 1
    print(i, c, a, c / a)
a, c = 0, 0
for k in datas_hard[0].keys():
    correct = False
    for i in range(files_num):
        if "gpt_eval" in datas_hard[i][k] and datas_hard[i][k]["gpt_eval"]:
            correct = True
    if correct:
        c += 1
    a += 1
print(c, a, c / a)

# a, c = 0, 0
# a1, c1 = 0, 0
# with open("/mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/data/agent_out/checkpoint-230_new.jsonl") as f:
#     for line in f.readlines():
#         data = json.loads(line)
#         if "yes" in data["gpt_result"].lower():
#             c += 1
#         a += 1
#         if not data["flag"]:
#             if "yes" in data["gpt_result"].lower():
#                 c1 += 1
#             a1 += 1
# print(c, a, c / a, c / 504)
# print(c1, a1, c1 / a1, c1 / 340)