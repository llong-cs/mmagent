import cv2
import base64
import os
import tempfile
from moviepy import VideoFileClip
from pydub import AudioSegment
from io import BytesIO
import numpy as np

def get_video_info(video_path):
    video = VideoFileClip(video_path)
    video_info = {}
    video_info["fps"] = video.fps
    video_info["frames"] = int(video.duration * video.fps)
    video_info["duration"] = video.duration
    video_info["path"] = video_path
    video_info["name"] = video_path.split("/")[-1]
    video_info["width"] = video.size[0]
    video_info["height"] = video.size[1]
    video_info["codec"] = None  # moviepy doesn't expose codec info
    video_info["format"] = None  # moviepy doesn't expose format info
    video_info["fourcc"] = None  # moviepy doesn't expose fourcc info
    video.close()
    return video_info

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


def get_video_codec(video_path):
    video = VideoFileClip(video_path)
    # Note: moviepy doesn't expose codec info directly
    video.close()
    return None


def process_video_clip(video_path, start_time, interval, fps=10):
    try:
        base64_data = {}
        video = VideoFileClip(video_path)

        # Ensure start_time + interval doesn't exceed video duration
        end_time = min(start_time + interval, video.duration)
        
        # Create subclip
        clip = video.subclipped(start_time, end_time)

        # Create temporary files
        temp_files = {
            "video": tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"),
            "audio": tempfile.NamedTemporaryFile(delete=False, suffix=".mp3"),
        }
        temp_paths = {k: f.name for k, f in temp_files.items()}
        for f in temp_files.values():
            f.close()

        # Write video without logging
        clip.write_videofile(temp_paths["video"], codec="libx264", audio_codec="aac", logger=None)

        # Write audio without logging 
        clip.audio.write_audiofile(temp_paths["audio"], codec="libmp3lame", logger=None)

        # Read files and convert to Base64
        for key, path in temp_paths.items():
            with open(path, "rb") as f:
                base64_data[key] = base64.b64encode(f.read()).decode("utf-8")
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


def get_audio_info_from_base64(base64_string, format_hint="mp3"):
    try:
        # decoding base64
        audio_data = base64.b64decode(base64_string)
        audio_io = BytesIO(audio_data)

        if format_hint:
            audio = AudioSegment.from_file(audio_io, format=format_hint)
        else:
            audio = AudioSegment.from_file(audio_io)

        duration = len(audio) / 1000  # ms to s
        channels = audio.channels
        frame_rate = audio.frame_rate
        sample_width = audio.sample_width

        return {
            "duration_seconds": duration,
            "channels": channels,
            "frame_rate_hz": frame_rate,
            "sample_width_bytes": sample_width,
        }
    except Exception as e:
        return {"error": str(e)}


def get_video_info_from_base64(base64_string):
    try:
        video_data = base64.b64decode(base64_string)

        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
            temp_video.write(video_data)
            temp_video.flush()

            video = VideoFileClip(temp_video.name)
            video_info = {
                "fps": video.fps,
                "frames": int(video.duration * video.fps),
                "duration": video.duration,
                "path": temp_video.name,
                "name": os.path.basename(temp_video.name),
                "width": video.size[0],
                "height": video.size[1],
                "codec": None,
                "format": None,
                "fourcc": None,
            }
            video.close()
            return video_info

    except Exception as e:
        return {"error": str(e)}