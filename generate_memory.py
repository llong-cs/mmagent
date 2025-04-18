import numpy as np
from tqdm import tqdm
import os
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from mmagent.videograph import VideoGraph
from mmagent.utils.general import *
from mmagent.utils.video_processing import *
from mmagent.utils.chat_api import *
from mmagent.prompts import *

from mmagent.face_processing import process_faces
from mmagent.voice_processing import process_voices
from mmagent.memory_processing import (
    process_captions,
    generate_captions_and_thinkings_with_ids,
)

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
        print("Finish processing voices")

    if not preprocessing or "face" in preprocessing:
        id2faces = process_faces(
            video_graph,
            base64_frames,
            save_path=os.path.join(save_path, f"clip_{clip_id}_faces.json"),
            preprocessing=preprocessing,
        )
        print("Finish processing faces")

    if preprocessing:
        print("Finish preprocessing segment")
        return

    episodic_captions, semantic_captions = generate_captions_and_thinkings_with_ids(
        video_graph,
        base64_video,
        base64_frames,
        id2faces,
        id2voices,
        clip_id,
    )

    process_captions(video_graph, episodic_captions, clip_id, type="episodic")
    process_captions(video_graph, semantic_captions, clip_id, type="semantic")

    print("Finish processing segment")


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

            print("=" * 20)

            print(f"Starting processing {clip_id}-th (out of {math.ceil(video_info['duration'] / interval_seconds)}) clip starting at {start_time} seconds...")
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
            # if clip_id < 44:
            #     continue
            print("=" * 20)
            full_path = os.path.join(video_path, video_file)
            print(f"Starting processing {clip_id}-th (out of {len(video_files)}) clip: {full_path}")

            base64_video, base64_frames, base64_audio = process_video_clip(
                video_path=full_path, start_time=0, interval=None, fps=fps, audio_format="wav"
            )

            # print("Start processing segment")

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
    
if __name__ == "__main__":
    # video paths can be paths to directories or paths to mp4 files
    data_list = ["CZ_1", "CZ_2", "CZ_3", "ZZ_1", "ZZ_2", "ZZ_3","ZZ_4", "ZZ_5"]
    # data_list = ["CZ_2"]
    # preprocessing = ['voice', 'face']
    preprocessing = []
    for data in data_list:
        input_dir = os.path.join(processing_config["input_dir"], data)
        video_files = os.listdir(input_dir)
        video_paths = [os.path.join(input_dir, video_file) for video_file in video_files]

        save_dir = os.path.join(processing_config["save_dir"], data)
        os.makedirs(save_dir, exist_ok=True)
        generated_memories = os.listdir(save_dir)
        generated_memories = [generated_memory for generated_memory in generated_memories if generated_memory.endswith(".pkl")]
        video_paths = [video_path for video_path in video_paths if generate_file_name(video_path)+".pkl" not in generated_memories]
        
        # save_dir = processing_config["save_dir"]
        # video_paths = ['/mnt/hdfs/foundation/longlin.kylin/mmagent/data/video_clips/EodRBU-HVEI']
        
        cpu_count = multiprocessing.cpu_count()
        # max_workers = min(cpu_count, processing_config.get("max_parallel_videos", 4))
        max_workers = 32
        
        print(f"Using {max_workers} processes (CPU cores: {cpu_count})")

        def process_single_video(args):
            video_path, save_dir = args
            video_graph = VideoGraph(**memory_config)
            try:
                streaming_process_video(video_graph, video_path, save_dir, preprocessing=preprocessing)
            except Exception as e:
                log_dir = processing_config["log_dir"]
                os.makedirs(log_dir, exist_ok=True)
                with open(os.path.join(log_dir, f"generate_memory_error.log"), "a") as f:
                    f.write(f"Error processing video {video_path}: {e}\n")

        args = [(video_path, save_dir) for video_path in video_paths]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process_single_video, args), total=len(video_paths), desc="Processing videos"))