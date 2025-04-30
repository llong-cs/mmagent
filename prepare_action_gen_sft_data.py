import json
from mmagent.prompts import prompt_refine_final_reasoning
from mmagent.utils.chat_api import generate_messages, parallel_get_response

def filter_data(input_file, output_file, max_samples_per_video=None):
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    video_ids = list(set([d["video_id"] for d in data]))

    selected_data = []

    for video_id in video_ids:
        video_data = [
            d
            for d in data
            if d["video_id"] == video_id and d["verify_result"].lower().startswith("yes")
        ]
        if max_samples_per_video is not None:
            video_data = video_data[:max_samples_per_video]
        for d in video_data:
            d["session"][1][-1]["action_content"] = d["answer"]
        selected_data.extend(video_data)

    with open(output_file.format(count=len(selected_data)), "w") as f:
        for d in selected_data:
            f.write(json.dumps(d) + "\n")

    print(f"Selected {len(selected_data)} samples")

def refine_final_reasoning(input_file):
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]
    data = [sample for sample in data if sample["session"] is not None]
    # for sample in data:
    #     try:
    #         test = sample[0]["session"][1][-1]["reasoning"]
    #     except Exception as e:
    #         print(e)
    #         print(sample)
    #         raise e
    inputs = [
        [
            {
                "type": "text",
                "content": prompt_refine_final_reasoning.format(reasoning=sample["session"][1][-1]["reasoning"])
            }
        ] for sample in data if len(sample["session"][1]) == 10
    ]
    messages = [generate_messages(input) for input in inputs]
    model = "gpt-4o-2024-11-20"
    responses = parallel_get_response(model, messages)[0]

    print(len(responses))

    idx = 0
    for sample in data:
        if len(sample["session"][1]) != 10:
            continue
        response = responses[idx]
        idx += 1
        sample["session"][1][-1]["reasoning"] = response
    print(idx)

    with open(input_file, "w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    filter_data("data/annotations/sft/0424/sft_0424.jsonl", "data/annotations/sft/0424/sft_0424_selected_{count}.jsonl", max_samples_per_video=4)
    # refine_final_reasoning("data/annotations/sft/0424/sft_0424.jsonl")
