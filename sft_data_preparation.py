import json

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

if __name__ == "__main__":
    filter_data("data/annotations/sft/0424/sft_0424.jsonl", "data/annotations/sft/0424/sft_0424_selected_{count}.jsonl", max_samples_per_video=4)
