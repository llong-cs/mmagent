import base64
import logging
import os
import tempfile
import math
import cv2
import numpy as np
from moviepy import VideoFileClip
from tqdm import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor
import shutil

# Disable moviepy logging
logging.getLogger('moviepy').setLevel(logging.ERROR)
# Disable moviepy's tqdm progress bar
logging.getLogger('moviepy.video.io.VideoFileClip').setLevel(logging.ERROR)
logging.getLogger('moviepy.audio.io.AudioFileClip').setLevel(logging.ERROR)

# Configure logging
logger = logging.getLogger(__name__)

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
    try: 
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
            with tempfile.NamedTemporaryFile(dir="data/temp", suffix=f".{video_format}") as temp_video:
            # with tempfile.NamedTemporaryFile(dir="data/videos", suffix=f".{video_format}") as temp_video:
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
        with tempfile.NamedTemporaryFile(dir="data/temp", suffix=f".{audio_format}") as temp_audio:
        # with tempfile.NamedTemporaryFile(dir="data/audios", suffix=f".{audio_format}") as temp_audio:
            # Check if audio exists
            if clip.audio is None:
                base64_data["audio"] = None
            else:
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
        logger.error(f"Error processing video clip: {str(e)}")
        raise

def split_video_into_clips(video_path, interval, output_dir, output_format='mp4'):
    """
    Split a video into clips of specified interval length and save them to a folder.
    
    Args:
        video_path (str): Path to the video file
        interval (int): Length of each clip in seconds
        output_dir (str): Directory to save the clips
        output_format (str): Format of the output clips (default: 'mp4')
        
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
        duration = int(video_info["duration"])

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
        
        def process_clip(clip_info):
            i, start_time, end_time = clip_info
            try:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(dir="data/temp", suffix=f".{output_format}", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # 创建子片段
                with VideoFileClip(video_path) as video:
                    clip = video.subclipped(start_time, end_time)
                    try:
                        # 写入临时文件
                        clip.write_videofile(temp_path, codec=video_codec, audio_codec=audio_codec, logger=None, threads=4)
                        # 移动到最终位置
                        output_path = os.path.join(output_dir, f"{i}.{output_format}")
                        shutil.move(temp_path, output_path)
                    finally:
                        clip.close()
            except Exception as e:
                # 清理临时文件
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass
                raise e
        
        # 准备所有片段的信息
        clip_infos = [(i, i * interval, min((i + 1) * interval, duration)) 
                      for i in range(num_clips)]
        
        # 使用线程池处理片段
        # 使用较少的线程，因为MoviePy内部已经使用了多线程
        internal_threads = min(4, num_clips)
        with ThreadPoolExecutor(max_workers=internal_threads) as executor:
            list(tqdm(executor.map(process_clip, clip_infos), total=num_clips, desc="Processing clips"))

        return output_dir

    except Exception as e:
        logger.error(f"Error splitting video into clips: {str(e)}")
        raise

def verify_video_processing(video_path, output_dir, interval, strict=False):
    """Verify that a video was properly split into clips by checking the number of clips.
    
    Args:
        video_path (str): Path to original video file
        output_dir (str): Directory containing the split clips
        interval (float): Interval length in seconds used for splitting
        
    Returns:
        bool: True if verification passes, False otherwise
    """

    def has_video_and_audio(file_path):
        def has_stream(stream_type):
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", stream_type,
                "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
                capture_output=True, text=True
            )
            return bool(result.stdout.strip())

        return has_stream("v:0") and has_stream("a:0")

    def has_static_segment(
        video_path,
        min_static_duration=5.0,  # 秒，静止时间阈值
        diff_threshold=0.001,  # 均值帧差小于该值就视为静止
    ) -> bool:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        min_static_frames = int(min_static_duration * fps)

        prev_gray = None
        consecutive_static_frames = 0

        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = np.mean(diff)

                if mean_diff < diff_threshold:
                    consecutive_static_frames += 1
                    if consecutive_static_frames >= min_static_frames:
                        cap.release()
                        return True
                else:
                    consecutive_static_frames = 0

            prev_gray = gray

        cap.release()
        return False

    try:
        if not os.path.exists(video_path):
            with open("logs/video_processing_failed.log", "a") as f:
                f.write(f"Error processing {video_path}: Video file not found.\n")
            logger.error(f"Error processing {video_path}: Video file not found.")
            return False
        # Get expected number of clips based on video duration
        video_info = get_video_info(video_path)
        expected_clips_num = math.ceil(int(video_info["duration"]) / interval)
        
        # Get actual number of clips in output directory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        clip_dir = os.path.join(output_dir, video_name)
        
        if not os.path.exists(clip_dir):
            with open("logs/video_processing_failed.log", "a") as f:
                f.write(f"Error processing {video_path}: Clip directory {clip_dir} not found.\n")
            logger.error(f"Error processing {video_path}: Clip directory {clip_dir} not found.")
            return False
            
        actual_clips = [f for f in os.listdir(clip_dir) if os.path.isfile(os.path.join(clip_dir, f)) and f.split('.')[-1] in ['mp4', 'mov', 'webm']]
        actual_clips_num = len(actual_clips)
        
        if actual_clips_num != expected_clips_num:
            with open("logs/video_processing_failed.log", "a") as f:
                f.write(f"Error processing {video_path}: Expected {video_info['duration']}/{interval}={expected_clips_num} clips, but found {actual_clips_num} clips.\n")
            logger.error(f"Error processing {video_path}: Expected {video_info['duration']}/{interval}={expected_clips_num} clips, but found {actual_clips_num} clips.")
            return False

        if strict:
            clip_files = [os.path.join(clip_dir, clip) for clip in actual_clips]
            for clip_file in clip_files:
                clip_id = clip_file.split("/")[-1].split(".")[0]
                if not has_video_and_audio(clip_file):
                    with open("logs/video_processing_failed.log", "a") as f:
                        f.write(f"Error processing {clip_file}: No video or audio streams found.\n")
                    logger.error(f"Error processing {clip_file}: No video or audio streams found.")
                    return False
                if int(clip_id) < len(clip_files)-2 and has_static_segment(clip_file):
                    with open("logs/video_processing_failed.log", "a") as f:
                        f.write(f"Error processing {clip_file}: Has static segment.\n")
                    logger.error(f"Error processing {clip_file}: Has static segment.")
                    return False
           
        return True
        
    except Exception as e:
        with open("logs/video_processing_failed.log", "a") as f:
            f.write(f"Error verifying {video_path}: {e}\n")
        logger.error(f"Error verifying {video_path}: {e}")
        return False

        # verify_video_processing("/mnt/hdfs/foundation/longlin.kylin/mmagent/data/raw_videos/ZZ_3/vdkQWgZLrYA.mp4", output_dir, interval)
