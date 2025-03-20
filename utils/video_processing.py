import cv2
import base64
import os
import tempfile
from moviepy import VideoFileClip
from pydub import AudioSegment
import numpy as np

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


def process_video_clip(video_path, start_time, interval, fps=10, video_format="mp4", audio_format="mp3"):
    try:
        base64_data = {}
        video = VideoFileClip(video_path)

        # Ensure start_time + interval doesn't exceed video duration
        end_time = min(start_time + interval, video.duration)
        
        # Create subclip
        clip = video.subclipped(start_time, end_time)

        # Create temporary files
        temp_files = {
            "video": tempfile.NamedTemporaryFile(delete=False, suffix=f".{video_format}"),
            "audio": tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}"),
        }
        temp_paths = {k: f.name for k, f in temp_files.items()}
        for f in temp_files.values():
            f.close()

        # Write video without logging
        if video_format == "mp4":
            video_codec = "libx264"
        elif video_format == "webm":
            video_codec = "libvpx"
        else:
            video_codec = "libx264"  # Default to mp4/h264
            
        clip.write_videofile(temp_paths["video"], codec=video_codec, audio_codec="aac", logger=None)

        # Write audio without logging
        if audio_format == "mp3":
            audio_codec = "libmp3lame"
        elif audio_format == "wav":
            audio_codec = "pcm_s16le"
        else:
            audio_codec = "libmp3lame"  # Default to mp3
            
        clip.audio.write_audiofile(temp_paths["audio"], codec=audio_codec, logger=None)

        # Read files and convert to Base64
        for key, path in temp_paths.items():
            with open(path, "rb") as f:
                base64_data[key] = base64.b64encode(f.read()).decode("utf-8")
                # base64_data[key] = base64.b64encode(f.read())
            os.remove(path)

        # Extract frames using adjusted interval
        actual_interval = end_time - start_time
        base64_data["frames"] = extract_frames(video_path, start_time, actual_interval, fps)

        video.close()
        clip.close()

        return base64_data["video"], base64_data["frames"], base64_data["audio"]

    except Exception as e:
        print(f"Error processing video clip: {str(e)}")
        raise


def get_audio_info_from_base64(base64_string):
    try:
        audio_data = base64.b64decode(base64_string)

        # Try common audio extensions
        for ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac']:
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