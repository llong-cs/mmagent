import numpy as np
from tqdm import tqdm
import os
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import logging
import argparse

from mmagent.videograph import VideoGraph
from mmagent.utils.general import *
from mmagent.utils.video_processing import *
from mmagent.utils.chat_api import *
from mmagent.prompts import *

from mmagent.face_processing import process_faces
from mmagent.voice_processing import process_voices
from mmagent.memory_processing import (
    process_memories,
    generate_memories,
)

# Configure logger
logger = logging.getLogger(__name__)

processing_config = json.load(open("configs/processing_config.json"))
memory_config = json.load(open("configs/memory_config.json"))

def process_segment(
    video_graph,
    base64_video,
    base64_frames,
    base64_audio,
    clip_id,
    video_path,
    preprocessing=[],
):
    save_path = os.path.join(
        processing_config["intermediate_save_dir"], generate_file_name(video_path)
    )

    if not preprocessing or "voice" in preprocessing:
        id2voices = process_voices(
            video_graph,
            base64_audio,
            base64_video,
            save_path=os.path.join(save_path, f"clip_{clip_id}_voices.json"),
            preprocessing=preprocessing,
        )
        logger.info("Finish processing voices")

    if not preprocessing or "face" in preprocessing:
        id2faces = process_faces(
            video_graph,
            base64_frames,
            save_path=os.path.join(save_path, f"clip_{clip_id}_faces.json"),
            preprocessing=preprocessing,
        )
        logger.info("Finish processing faces")

    if preprocessing:
        logger.info("Finish preprocessing segment")
        return

    episodic_memories, semantic_memories = generate_memories(
        base64_video,
        base64_frames,
        id2faces,
        id2voices,
    )

    process_memories(video_graph, episodic_memories, clip_id, type="episodic")
    process_memories(video_graph, semantic_memories, clip_id, type="semantic")

    logger.info("Finish processing segment")


def streaming_process_video(video_graph, video_path, save_dir, preprocessing=[]):
    """Process video segments at specified intervals with given fps.

    Args:
        video_graph (VideoGraph): Graph object to store video information
        video_path (str): Path to the video file or directory containing clips
        interval_seconds (float): Time interval between segments in seconds
        fps (float): Frames per second to extract from each segment

    Returns:
        None: Updates video_graph in place with processed segments
    """
    interval_seconds = processing_config["interval_seconds"]
    fps = processing_config["fps"]
    segment_limit = processing_config["segment_limit"]

    if os.path.isfile(video_path):
        # Process single video file
        video_info = get_video_info(video_path)

        # Process each interval
        clip_id = 0
        for start_time in np.arange(0, video_info["duration"], interval_seconds):
            if start_time + interval_seconds > video_info["duration"]:
                break

            logger.info("=" * 20)

            logger.info(f"Starting processing {clip_id}-th (out of {math.ceil(video_info['duration'] / interval_seconds)}) clip starting at {start_time} seconds...")
            base64_video, base64_frames, base64_audio = process_video_clip(
                video_path, start_time, interval_seconds, fps, audio_format="wav"
            )

            # Process frames for this interval
            if base64_frames:
                process_segment(
                    video_graph,
                    base64_video,
                    base64_frames,
                    base64_audio,
                    clip_id,
                    video_path,
                    preprocessing,
                )

            clip_id += 1

            if segment_limit > 0 and clip_id >= segment_limit:
                break

    elif os.path.isdir(video_path):
        # Process directory of numbered clips
        files = os.listdir(video_path)
        # Filter for video files and sort by numeric value in filename
        video_files = [
            f for f in files if any(f.endswith(ext) for ext in [".mp4", ".avi", ".mov"])
        ]
        video_files.sort(key=lambda x: int("".join(filter(str.isdigit, x))))

        for clip_id, video_file in enumerate(video_files):
            if segment_limit > 0 and clip_id >= segment_limit:
                break
            logger.info("=" * 20)
            full_path = os.path.join(video_path, video_file)
            logger.info(f"Starting processing {clip_id}-th (out of {len(video_files)}) clip: {full_path}")

            base64_video, base64_frames, base64_audio = process_video_clip(
                video_path=full_path, start_time=0, interval=None, fps=fps, audio_format="wav"
            )

            if base64_frames:
                process_segment(
                    video_graph,
                    base64_video,
                    base64_frames,
                    base64_audio,
                    clip_id,
                    video_path,
                    preprocessing,
                )
    
    if preprocessing:
        return
    
    video_graph.refresh_equivalences()
        
    save_video_graph(
        video_graph,
        video_path, 
        save_dir
    )
    
def process_single_video(args):
    video_path, save_dir = args
    video_graph = VideoGraph(**memory_config)
    try:
        streaming_process_video(video_graph, video_path, save_dir, preprocessing=preprocessing)
    except Exception as e:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"generate_memory_error.log"), "a") as f:
            f.write(f"Error processing video {video_path}: {e}\n")
        logger.error(f"Error processing video {video_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessing", type=str, default="voice,face")
    parser.add_argument("--data_list", type=str, default="/mnt/bn/videonasi18n/longlin.kylin/mmagent/data/annotations/small_test.jsonl")
    parser.add_argument("--log_dir", type=str, default="data/logs")
    args = parser.parse_args()
    log_dir = args.log_dir

    data_list = args.data_list.split(',')
    preprocessing = args.preprocessing.split(',')
    if len(preprocessing) == 0:
        preprocessing = []
    
    
    args = []
    
    for file in data_list:
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                if not os.path.exists(sample["mem_path"]):
                    save_dir = os.path.dirname(sample["mem_path"])
                    clips_dir = sample["clip_path"]
                    os.makedirs(save_dir, exist_ok=True)
                    args.append((clips_dir, save_dir))
    
    args = list(set(args))
        
    cpu_count = multiprocessing.cpu_count()
    max_workers = 32
    
    logger.info(f"Total video inputs: {len(args)}")
    logger.info(f"Using {max_workers} processes (CPU cores: {cpu_count})")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_single_video, args), total=len(args), desc="Processing videos"))