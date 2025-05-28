import json
import os
from mmagent.utils.general import generate_file_name

clips_dir = "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/video_clips"
mems_dir = "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/mems"

def prepare_video_mme_data():
    outputs = []
    clips = os.listdir(os.path.join(clips_dir, "Video-MME"))
    videos_dir = "/mnt/hdfs/foundation/longlin.kylin/mmagent/benchmarks/Video-MME/videos"
    videos = os.listdir(videos_dir)
    with open("/mnt/hdfs/foundation/longlin.kylin/mmagent/benchmarks/Video-MME/test.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            if data["duration"] == "long":
                video_id = data["videoID"]
                if video_id in clips and video_id+".mp4" in videos:
                    video_path = os.path.join(videos_dir, video_id+".mp4")
                    clip_path = os.path.join(clips_dir, "Video-MME", video_id)
                    mem_path = os.path.join(mems_dir, "Video-MME", generate_file_name(video_path)+".pkl")
                    outputs.append({
                        "video_id": data["video_id"],
                        "video_url": data["url"],
                        "video_path": video_path,
                        "clip_path": clip_path,
                        "mem_path": mem_path,
                        "question": data["question"] + "\nOptions: " + " ".join(data["options"]),
                        "answer": data["answer"],
                    })
    print(len(outputs))
    with open("data/benchmarks/Video-MME.json", "w") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
    for output in outputs:
        with open("data/benchmarks/Video-MME.jsonl", "a") as f:
            f.write(json.dumps(output, ensure_ascii=False)+"\n")
            
def prepare_mlvu_data():
    data_dir = "/mnt/hdfs/foundation/longlin.kylin/mmagent/benchmarks/MLVU/MLVU/json"
    videos_dir = "/mnt/hdfs/foundation/longlin.kylin/mmagent/benchmarks/MLVU/MLVU/video"
    data_list = os.listdir(data_dir)
    outputs = []
    for file in data_list:
        marker = file.split(".")[0]
        file_path = os.path.join(data_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            for sample in data:
                video_id = sample["video"].split(".")[0]
                video_path = os.path.join(videos_dir, marker, sample["video"])
                clip_path = os.path.join(clips_dir, "MLVU", marker, video_id)
                mem_path = os.path.join(mems_dir, "MLVU", generate_file_name(video_path)+".pkl")
                if os.path.exists(video_path):
                    if sample["question_type"] not in ["subPlot", "summary"]:
                        outputs.append({
                            "video_id": video_id,
                            "video_path": video_path,
                            "clip_path": clip_path,
                            "mem_path": mem_path,
                            "question": sample["question"] + "\nOptions: " + " ".join(sample["candidates"]),
                            "answer": sample["answer"],
                        })
                    else:
                        outputs.append({
                            "video_id": video_id,
                            "video_path": video_path,
                            "clip_path": clip_path,
                            "mem_path": mem_path,
                            "question": sample["question"],
                            "answer": sample["answer"],
                        })
    print(len(outputs))
    with open("data/benchmarks/MLVU.json", "w") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
    for output in outputs:
        with open("data/benchmarks/MLVU.jsonl", "a") as f:
            f.write(json.dumps(output, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    # prepare_video_mme_data()
    prepare_mlvu_data()
