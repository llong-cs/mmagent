import logging
import json
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

from mmagent.utils.video_processing import split_video_into_clips, verify_video_processing

# Disable moviepy logging
logging.getLogger('moviepy').setLevel(logging.ERROR)
# Disable moviepy's tqdm progress bar
logging.getLogger('moviepy.video.io.VideoFileClip').setLevel(logging.ERROR)
logging.getLogger('moviepy.audio.io.AudioFileClip').setLevel(logging.ERROR)

# Configure logging
logger = logging.getLogger(__name__)

processing_config = json.load(open("configs/processing_config.json"))
interval = processing_config["interval_seconds"]
log_dir = processing_config["log_dir"]

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
        return path
    return None

if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    max_workers = min(cpu_count, 20)
    logger.info(f"Using {max_workers} processes (CPU cores: {cpu_count})")

    # annotations_paths = ["data/annotations/ZZ_4_refined.json", "data/annotations/ZZ_5_refined.json"]
    
    # for annotations_path in annotations_paths:
    #     marker = annotations_path.split("/")[-1].split(".")[0].strip("_refined")
    #     with open(annotations_path, "r") as f:
    #         videos = json.load(f)
    #     output_dir = os.path.join("/mnt/hdfs/foundation/longlin.kylin/mmagent/data/video_clips", marker)
    #     os.makedirs(output_dir, exist_ok=True)

    #     # Check video paths in parallel
    #     with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #         args = [(video["path"], output_dir, interval) for video in videos]
    #         paths = list(tqdm(executor.map(check_video_path, args), total=len(args), desc="Checking video paths"))
    #         video_paths = [path for path in paths if path is not None]

    #     # Process videos in parallel
    #     with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #         args = [(video_path, interval, output_dir) for video_path in video_paths]
    #         list(tqdm(executor.map(process_video_parallel, args), total=len(args), desc="Processing videos"))
        
    #     # Verify all videos in parallel
    #     with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #         args = [(video["path"], output_dir, interval) for video in videos if os.path.exists(video["path"])]
    #         list(tqdm(executor.map(verify_video_parallel, args), total=len(args), desc="Verifying videos"))
        
    # 
    

    base_dir = "/mnt/hdfs/foundation/longlin.kylin/mmagent/benchmarks/Video-MME/videos"
    all_videos = os.listdir(base_dir)
    all_videos = [os.path.join(base_dir, video) for video in all_videos]
    nodes = 2
    index = 1
    batch_size = len(all_videos) // nodes
    
    for i in range(nodes):
        if i != index:
            continue
        videos = all_videos[i*batch_size:(i+1)*batch_size]
        
        output_dir = "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/video_clips/Video-MME"
        os.makedirs(output_dir, exist_ok=True)

        # Check video paths in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            args = [(video, output_dir, interval) for video in videos]
            paths = list(tqdm(executor.map(check_video_path, args), total=len(args), desc="Checking video paths"))
            todo_video_paths = [path for path in paths if path is not None]

        # Process videos in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            args = [(video_path, interval, output_dir) for video_path in todo_video_paths]
            list(tqdm(executor.map(process_video_parallel, args), total=len(args), desc="Processing videos"))
        
        # Verify all videos in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            args = [(video, output_dir, interval) for video in videos if os.path.exists(video)]
            list(tqdm(executor.map(verify_video_parallel, args), total=len(args), desc="Verifying videos"))