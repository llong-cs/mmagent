import logging
import json
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse
from tqdm import tqdm

from mmagent.utils.video_processing import split_video_into_clips, verify_video_processing

# Disable moviepy logging
logging.getLogger('moviepy').setLevel(logging.ERROR)
# Disable moviepy's tqdm progress bar
logging.getLogger('moviepy.video.io.VideoFileClip').setLevel(logging.ERROR)
logging.getLogger('moviepy.audio.io.AudioFileClip').setLevel(logging.ERROR)

# Configure logging
logger = logging.getLogger(__name__)


def process_video_parallel(args):
    video_path, interval, output_dir = args
    try:
        split_video_into_clips(video_path, interval, output_dir)
    except Exception as e:
        logger.error(f"Error processing {video_path}: {str(e)}")

def verify_video_parallel(args):
    video_path, output_dir, interval = args
    if not verify_video_processing(video_path, output_dir, interval, strict=True):
        with open(os.path.join(log_dir, f"video_processing_error.log"), "a") as f:
            f.write(video_path + "\n")
        return False
    return True

def check_video_path(args):
    path, output_dir, interval = args
    if not verify_video_processing(path, output_dir, interval, strict=True):
        return (path, output_dir, interval)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, default="/mnt/bn/videonasi18n/longlin.kylin/mmagent/data/annotations/train_500.jsonl,/mnt/bn/videonasi18n/longlin.kylin/mmagent/data/annotations/full_test.jsonl")
    parser.add_argument("--machine_number", type=int, default=1)
    parser.add_argument("--machine_index", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="data/logs")
    args = parser.parse_args()
    processing_config = json.load(open("configs/processing_config.json"))
    interval = processing_config["interval_seconds"]
    log_dir = args.log_dir
        
    input_data = args.input_data.split(",")
    machine_number = args.machine_number
    machine_index = args.machine_index
    
    cpu_count = multiprocessing.cpu_count()
    max_workers = cpu_count // 2
    logger.info(f"Using {max_workers} processes (CPU cores: {cpu_count})")
    
    all_videos = []
    for data_path in input_data:
        with open(data_path, "r") as f:
            for line in f:
                data = json.loads(line)
                video_path = data["video_path"]
                output_dir = data["clip_path"]
                if (video_path, output_dir, interval) not in all_videos:
                    os.makedirs(output_dir, exist_ok=True)
                    all_videos.append((video_path, output_dir, interval))

    all_videos.sort(key=lambda x: x[0])
    all_videos = all_videos[machine_index::machine_number]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        todo_videos = list(tqdm(executor.map(check_video_path, all_videos), total=len(all_videos), desc="Checking video paths"))
        todo_videos = [video for video in todo_videos if video is not None]
    
    for video_path, output_dir, interval in todo_videos:
        if not verify_video_processing(video_path, output_dir, interval, strict=True):
            with open(os.path.join(log_dir, f"video_processing_error.log"), "a") as f:
                f.write(video_path + "\n")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_video_parallel, todo_videos), total=len(todo_videos), desc="Processing videos"))
        
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(verify_video_parallel, todo_videos), total=len(todo_videos), desc="Verifying videos"))