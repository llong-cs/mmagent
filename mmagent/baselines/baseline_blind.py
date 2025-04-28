import json
from utils.chat_api import *
from qa_processing import *

qa_answered_blindly = []

with open("data/annotations/small_test.jsonl", "r") as f:
    inputs = [
        [
            {
                "type": "text",
                "content": json.loads(line)["question"]
            }
        ] for line in f
    ]
    print(len(inputs))
    messages = [generate_messages(input) for input in inputs]
    model = "gpt-4o-2024-08-06"
    responses = parallel_get_response(model, messages)
    agent_answers = responses[0]

with open("data/annotations/small_test.jsonl", "r") as f:
    idx = 0
    for line in f:
        qa = json.loads(line)
        qa["agent_answer"] = agent_answers[idx]
        qa_answered_blindly.append(qa)
        idx += 1

print(len(qa_answered_blindly))

with open("data/annotations/results/0416/baseline_blindly_answers.jsonl", "w") as f:
    for qa in qa_answered_blindly:
        f.write(json.dumps(qa) + "\n")