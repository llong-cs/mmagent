import base64
import logging
import os
import tempfile
import math
import cv2
import numpy as np
from moviepy import VideoFileClip
from pydub import AudioSegment
from tqdm import tqdm

# logging.getLogger('moviepy').setLevel(logging.ERROR)

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
    
    # Handle audio files
    audio_formats = ['mp3', 'wav', 'ogg', 'm4a']
    if file_info["format"] in audio_formats:
        audio = AudioSegment.from_file(file_path)
        file_info["duration"] = len(audio) / 1000.0  # Convert ms to seconds
        file_info["channels"] = audio.channels
        file_info["sample_width"] = audio.sample_width
        file_info["frame_rate"] = audio.frame_rate
        file_info["type"] = "audio"
        return file_info
        
    # Handle video files using moviepy
    video = VideoFileClip(file_path)
    
    # Get basic properties from moviepy
    file_info["fps"] = video.fps
    file_info["frames"] = int(video.fps * video.duration)
    file_info["duration"] = video.duration
    file_info["width"] = video.size[0]
    file_info["height"] = video.size[1]
    
    # Get codec info from moviepy if available
    try:
        file_info["codec"] = video.reader.infos['video_codec']
    except (KeyError, AttributeError):
        file_info["codec"] = "unknown"
        
    file_info["type"] = "video"
    
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


def get_audio_info_from_base64(base64_string):
    try:
        audio_data = base64.b64decode(base64_string)

        # Try common audio extensions
        for ext in ['.wav', '.mp3', '.m4a', '.ogg', '.flac']:
            try:
                with tempfile.NamedTemporaryFile(delete=True, suffix=ext) as temp_audio:
                    temp_audio.write(audio_data)
                    temp_audio.flush()
                    return get_video_info(temp_audio.name)
            except:
                continue

        raise Exception("Could not determine audio format")

    except Exception as e:
        return {"error": str(e)}


def get_video_info_from_base64(base64_string):
    try:
        video_data = base64.b64decode(base64_string)

        # Try common video extensions
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            try:
                with tempfile.NamedTemporaryFile(delete=True, suffix=ext) as temp_video:
                    temp_video.write(video_data)
                    temp_video.flush()
                    return get_video_info(temp_video.name)
            except:
                continue

        raise Exception("Could not determine video format")

    except Exception as e:
        return {"error": str(e)}

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
        # format = video_info["format"]

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
                    clip.write_videofile(output_path, codec=video_codec, audio_codec=audio_codec, logger=None, threads=4)

        return output_dir

    except Exception as e:
        print(f"Error splitting video into clips: {str(e)}")
        raise

if __name__ == "__main__":
    video_path = "/mnt/bn/videonasi18n/longlin.kylin/vlm-agent-benchmarking/data/videos/raw/720p/5 Poor People vs 1 Secret Millionaire.mp4"
    interval = 30
    output_dir = "data/videos/clipped"
    split_video_into_clips(video_path, interval, output_dir)
