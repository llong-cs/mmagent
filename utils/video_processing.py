import base64
import logging
import os
import tempfile
import math
import json
import cv2
import numpy as np
from moviepy import VideoFileClip
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import shutil
import contextlib
import io

# Disable moviepy logging
logging.getLogger('moviepy').setLevel(logging.ERROR)
# Disable moviepy's tqdm progress bar
logging.getLogger('moviepy.video.io.VideoFileClip').setLevel(logging.ERROR)
logging.getLogger('moviepy.audio.io.AudioFileClip').setLevel(logging.ERROR)

processing_config = json.load(open("configs/processing_config.json"))

def get_video_info(file_path):
    """Get video/audio information using appropriate libraries.
    
    Args:
        file_path (str): Path to video or audio file
        
    Returns:
        dict: Dictionary containing media metadata
    """
    file_info = {}
    file_info["path"] = file_path
    file_info["name"] = file_path.split("/")[-1]
    file_info["format"] = os.path.splitext(file_path)[1][1:].lower()
        
    # Handle video files using moviepy
    
    video = VideoFileClip(file_path)  # Disable logging for this instance
    
    # Get basic properties from moviepy
    file_info["fps"] = video.fps
    file_info["frames"] = int(video.fps * video.duration)
    file_info["duration"] = video.duration
    file_info["width"] = video.size[0]
    file_info["height"] = video.size[1]
    
    video.close()
    return file_info

def extract_frames(video_path, start_time=None, interval=None, sample_fps=10):
    video = VideoFileClip(video_path)

    # if start_time and interval are not provided, sample the whole video at sample_fps
    if start_time is None and interval is None:
        start_time = 0
        interval = video.duration

    frames = []
    frame_interval = 1.0 / sample_fps

    # Extract frames at specified intervals
    for t in np.arange(
        start_time, min(start_time + interval, video.duration), frame_interval
    ):
        frame = video.get_frame(t)
        # Convert frame to jpg and base64
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frames.append(base64.b64encode(buffer).decode("utf-8"))
        
    video.close()
    return frames

# TODO: check if there is a better way to do this without repeatedly opening and closing the video file
def process_video_clip(video_path, start_time, interval=None, fps=10, video_format="mp4", audio_format="wav", audio_fps=16000): 
    try                                                                                                                       : 
        base64_data = {}
        video = VideoFileClip(video_path)

        if interval is None:
            # Process entire video
            clip = video
            # Read video file directly
            with open(video_path, "rb") as f:
                base64_data["video"] = base64.b64encode(f.read())
        else:
            # Process subclip
            end_time = min(start_time + interval, video.duration)
            clip = video.subclipped(start_time, end_time)
            
            # Create temporary video file using context manager
            with tempfile.NamedTemporaryFile(dir="data/videos", suffix=f".{video_format}") as temp_video:
                # Determine codecs based on format
                if video_format in ['mp4', 'mov']:
                    video_codec = 'libx264'
                    audio_codec = 'aac'
                elif video_format == 'webm':
                    video_codec = 'libvpx'
                    audio_codec = 'libvorbis'
                else:
                    video_codec = 'libx264'  # Default to H.264
                    audio_codec = 'aac'
                
                clip.write_videofile(temp_video.name, codec=video_codec, audio_codec=audio_codec, logger=None, threads=4)

                # Read video file and convert to Base64
                temp_video.seek(0)
                base64_data["video"] = base64.b64encode(temp_video.read())

        # Create temporary audio file using context manager
        with tempfile.NamedTemporaryFile(dir="data/audios", suffix=f".{audio_format}") as temp_audio:
            # Write audio without logging, using specified fps for audio sampling
            if audio_format == "mp3":
                audio_codec = "libmp3lame"
            elif audio_format == "wav":
                audio_codec = "pcm_s16le"
            else:
                audio_codec = "libmp3lame"  # Default to mp3
            
            clip.audio.write_audiofile(temp_audio.name, codec=audio_codec, fps=audio_fps, logger=None)

            # Read audio file and convert to Base64
            temp_audio.seek(0)
            base64_data["audio"] = base64.b64encode(temp_audio.read())

        # Extract frames using adjusted interval
        if interval is None:
            base64_data["frames"] = extract_frames(video_path, sample_fps=fps)
        else:
            actual_interval = end_time - start_time
            base64_data["frames"] = extract_frames(video_path, start_time, actual_interval, sample_fps=fps)

        video.close()
        if interval is not None:
            clip.close()

        return base64_data["video"], base64_data["frames"], base64_data["audio"]

    except Exception as e:
        print(f"Error processing video clip: {str(e)}")
        raise

def split_video_into_clips(video_path, interval, output_dir, output_format='mp4'):
    """
    Split a video into clips of specified interval length and save them to a folder.
    
    Args:
        video_path (str): Path to the video file
        interval (int): Length of each clip in seconds
        
    Returns:
        str: Path to the output folder containing the clips
    """
    try:
        # Create output folder based on video filename
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_dir, video_name)
        os.makedirs(output_dir, exist_ok=True)

        # Get video info
        video_info = get_video_info(video_path)
        duration = video_info["duration"]

        # Determine codecs based on format
        if output_format in ['mp4', 'mov']:
            video_codec = 'libx264'
            audio_codec = 'aac'
        elif output_format == 'webm':
            video_codec = 'libvpx'
            audio_codec = 'libvorbis'
        else:
            video_codec = 'libx264'  # Default to H.264
            audio_codec = 'aac'

        # Calculate number of clips
        num_clips = math.ceil(duration / interval)

        # Open video using context manager
        for i in tqdm(range(num_clips)):
            start_time = i * interval
            end_time = min((i + 1) * interval, duration)
            # Create and process clip in its own context
            with VideoFileClip(video_path) as video:
                with video.subclipped(start_time, end_time) as clip:
                    output_path = os.path.join(output_dir, f"{i+1}.{output_format}")
                    # Use unique tempfile for each clip processing
                    with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    try:
                        # Write to temporary file first
                        clip.write_videofile(temp_path, codec=video_codec, audio_codec=audio_codec, logger=None, threads=4)
                        # Move to final location using shutil.move
                        shutil.move(temp_path, output_path)
                    except Exception as e:
                        # Clean up temp file if something goes wrong
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                        raise e

        return output_dir

    except Exception as e:
        print(f"Error splitting video into clips: {str(e)}")
        raise

def verify_video_processing(video_path, output_dir, interval):
    """Verify that a video was properly split into clips by checking the number of clips.
    
    Args:
        video_path (str): Path to original video file
        output_dir (str): Directory containing the split clips
        interval (float): Interval length in seconds used for splitting
        
    Returns:
        bool: True if verification passes, False otherwise
    """
    try:
        # Get expected number of clips based on video duration
        video_info = get_video_info(video_path)
        expected_clips = math.ceil(video_info["duration"] / interval)
        
        # Get actual number of clips in output directory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        clip_dir = os.path.join(output_dir, video_name)
        
        if not os.path.exists(clip_dir):
            print(f"Output directory {clip_dir} does not exist")
            return False
            
        actual_clips = len([f for f in os.listdir(clip_dir) if os.path.isfile(os.path.join(clip_dir, f)) and f.split('.')[-1] in ['mp4', 'mov', 'webm']])
        
        if actual_clips != expected_clips:
            print(f"Mismatch for {video_path}:")
            print(f"Expected {expected_clips} clips but found {actual_clips}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error verifying {video_path}: {str(e)}")
        return False


if __name__ == "__main__":
    annotations = json.load(open("data/annotations/video_list_CZ_answer_clipwise_0320.json"))
    video_paths = [video["path"] for video in annotations]
    video_paths = [os.path.join("/mnt/hdfs/foundation/longlin.kylin/mmagent/data/raw_videos", video_path.split("/")[-1]) for video_path in video_paths]
    interval = processing_config["interval_seconds"]
    output_dir = processing_config["input_dir"]
    
    def process_video_parallel(args):
        video_path, interval, output_dir = args
        try:
            split_video_into_clips(video_path, interval, output_dir)
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
    
    # Calculate optimal number of workers
    cpu_count = multiprocessing.cpu_count()
    # For I/O bound tasks like video processing, we can use slightly more workers than CPU cores
    # but we'll cap it to avoid system overload
    max_workers = min(cpu_count * 1.5, 32)  # Cap at 32 workers maximum
    max_workers = int(max_workers)  # Convert to integer
    
    print(f"Using {max_workers} workers (CPU cores: {cpu_count})")

    video_paths = ["data/videos/raw/test/test2.mp4"]
    output_dir = "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/test_video_clips"
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        args = [(video_path, interval, output_dir) for video_path in video_paths]
        list(tqdm(executor.map(process_video_parallel, args), total=len(args), desc="Processing videos"))
    
    # verify video processing
    for video_path in video_paths:
        if not verify_video_processing(video_path, output_dir, interval):
            print(f"Verification failed for {video_path}")
            break
