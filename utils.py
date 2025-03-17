import openai
import os
import cv2, base64
from tqdm import tqdm
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from time import sleep
import numpy as np
import hashlib
from datetime import datetime
from moviepy import *
import re


# api utils

config = json.load(open("api_config.json"))
client = {}
for model_name in config.keys():
    client[model_name] = openai.AzureOpenAI(
        azure_endpoint=config[model_name]["azure_endpoint"],
        api_version=config[model_name]["api_version"],
        api_key=config[model_name]["api_key"],
    )

MAX_RETRIES = 10

def get_response(model, messages):
    """Get chat completion response from specified model.

    Args:
        model (str): Model identifier
        messages (list): List of message dictionaries

    Returns:
        tuple: (response content, total tokens used)
    """
    response = client[model].chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    # return answer and number of tokens
    return response.choices[0].message.content, response.usage.total_tokens

def get_response_with_retry(model, messages):
    """Retry get_response up to MAX_RETRIES times with error handling.

    Args:
        model (str): Model identifier
        messages (list): List of message dictionaries

    Returns:
        tuple: (response content, total tokens used)
        
    Raises:
        Exception: If all retries fail
    """
    for i in range(MAX_RETRIES):
        try:
            return get_response(model, messages)
        except Exception as e:
            sleep(30)
            print(f"Retry {i} times, exception: {e}")
            continue
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")

def parallel_get_response(model, messages):
    """Process multiple messages in parallel using ThreadPoolExecutor.

    Args:
        model (str): Model identifier
        messages (list): List of message lists to process

    Returns:
        tuple: (list of responses, total tokens used)
    """
    max_workers = min(len(messages), config[model]["qpm"])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        responses = list(executor.map(lambda x: get_response_with_retry(model, x), messages))
    # answers are the first column of responses
    answers = [response[0] for response in responses]
    # tokens are the second column of responses 
    tokens = [response[1] for response in responses]
    # return answers and sum of tokens
    return answers, sum(tokens)


def get_embedding(model, text):
    """Get embedding for text using specified model.

    Args:
        model (str): Model identifier
        text (str): Text to embed

    Returns:
        tuple: (embedding vector, total tokens used)
    """
    return client[model].embeddings.create(input=text, model=model).data[0].embedding, client[model].embeddings.create(input=text, model=model).usage.total_tokens


def get_embedding_with_retry(model, text):
    """Retry get_embedding up to MAX_RETRIES times with error handling.

    Args:
        model (str): Model identifier
        text (str): Text to embed

    Returns:
        tuple: (embedding vector, total tokens used)
        
    Raises:
        Exception: If all retries fail
    """
    for i in range(MAX_RETRIES):
        try:
            return get_embedding(model, text)
        except Exception as e:
            sleep(30)
            print(f"Retry {i} times, exception: {e}")
            continue
    raise Exception(f"Failed to get embedding after {MAX_RETRIES} retries")

def parallel_get_embedding(model, texts):
    """Process multiple texts in parallel to get embeddings.

    Args:
        model (str): Model identifier
        texts (list): List of texts to embed

    Returns:
        tuple: (list of embeddings, total tokens used)
    """
    max_workers = min(len(texts), config[model]["qpm"])
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda x: get_embedding_with_retry(model, x), texts))
    # Split results into embeddings and tokens
    embeddings = [result[0] for result in results]
    tokens = [result[1] for result in results]
    # return embeddings and sum of tokens
    return embeddings, sum(tokens)

def get_whisper(model, file_path):
    """Transcribe audio file using Whisper model.

    Args:
        model (str): Model identifier
        file_path (str): Path to audio file

    Returns:
        str: Transcription text
    """
    file = open(file_path, "rb")
    return client[model].audio.transcriptions.create(model=model, file=file).text

def get_whisper_with_retry(model, file_path):
    """Retry Whisper transcription with error handling.

    Args:
        model (str): Model identifier
        file_path (str): Path to audio file

    Returns:
        str: Transcription text
        
    Raises:
        Exception: If all retries fail
    """
    for i in range(MAX_RETRIES):
        try:
            return get_whisper(model, file_path)
        except Exception as e:
            sleep(30)
            print(f"Retry {i} times, exception: {e}")
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")

def parallel_get_whisper(model, file_paths):
    """Process multiple audio files in parallel using Whisper model.

    Args:
        model (str): Model identifier
        file_paths (list): List of audio file paths

    Returns:
        list: List of transcription results
    """
    max_workers = min(len(file_paths), config[model]["qpm"])
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        responses = list(executor.map(lambda x: get_whisper_with_retry(model, x), file_paths))
    return responses

def generate_messages(inputs):
    """Generate message list for chat completion from mixed inputs.

    Args:
        inputs (list): List of input dictionaries with 'type' and 'content' keys
        type can be "text", "images", "video"
        content should be a string for text, 
        a list of base64 encoded images for images, 
        or a string (url) for video

    Returns:
        list: Formatted messages for chat completion
    """
    messages = []
    messages.append(
        {"role": "system", "content": "You are an expert in video understanding."}
    )
    content = []
    for input in inputs:
        if input["type"] == "text":
            content.append({"type": "text", "text": input["content"]})
        elif input["type"] == "images":
            if isinstance(input["content"][0], str):
                content.extend(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img}",
                                "detail": "high",
                            },
                        }
                        for img in input["content"]
                    ]
                )
            else:
                for img in input["content"]:
                    content.append({
                        "type": "text",
                        "text": img[0],
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img[1]}",
                            "detail": "high",
                        },
                    })
        elif input["type"] == "video_url":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": input["content"]},
                }
            )
        elif input["type"] == "video_base64":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:video/mp4;base64,{input['content']}"},
                }
            )        
        elif input["type"] == "audio":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:audio/mp3;base64,{input['content']}"
                    },
                }
            )
        else:
            raise ValueError(f"Invalid input type: {input['type']}")
    messages.append({"role": "user", "content": content})
    return messages

# video processing
def extract_frames_at_time(video_path, time_stamp):
    """Extract video frame at specified timestamp.

    Args:
        video_path (str): Path to video file
        time_stamp (float): Time in seconds to extract frame

    Returns:
        list: List containing single base64 encoded frame image
    """
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_MSEC, time_stamp * 1000)
    ret, frame = video.read()
    if not ret:
        print(f"Failed to extract frame at time {time_stamp}")
    _, buffer = cv2.imencode(".jpg", frame)
    return [base64.b64encode(buffer).decode("utf-8")]

def extract_frames(video_path, frame_interval):
    """Extract frames from video at specified intervals.

    Args:
        video_path (str): Path to video file
        frame_interval (float): Time interval between frames in seconds

    Returns:
        list: List of base64 encoded frame images
    """
    video = cv2.VideoCapture(video_path)  # type: ignore
    frames = []
    count = 0
    # get frame rate
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * frame_interval)
    # print(f"Frame rate: {frame_rate}")
    # print(f"Frame interval: {frame_interval}")
    while video.isOpened():
        success, frame = video.read()
        # print(f"Frame interval: {frame_interval}")
        if not success:
            break
        if count % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))
        count += 1
    video.release()
    return frames

# file processing
def get_video_paths(video_url, task):
    """Generate video and segment paths from URL and task.

    Args:
        video_url (str): URL of the video
        task (str): Task identifier

    Returns:
        tuple: (video save path, segment save path)
    """
    video_name = video_url.split("/")[-1].split(".")[0].split("_")[3]
    segment_name = video_url.split("/")[-1].split(".")[0]
    video_save_path = os.path.join(f"output/{task}", video_name)
    segment_save_path = os.path.join(video_save_path, segment_name)
    return video_save_path, segment_save_path

def get_video_names(path):
    """Extract unique video names in a given directory.

    Args:
        path (str): Directory path to search

    Returns:
        list: List of unique video names
    """
    files = os.listdir(path)
    video_names = [file.split("_")[3] for file in files]
    video_names = list(set(video_names))
    return video_names

def get_files_by_name(base_path, video_name, video_config):
    """Get video files matching the specified name and config.
    Files can be .mp4, .mp3, .srt, .txt, depending on the base_path type.

    Args:
        base_path (str): Directory path to search
        video_name (str): Video name to match
        video_config (dict): Video configuration with resolution, clip_size, and clip_duration

    Returns:
        list: Sorted list of matching video files
    """
    files = os.listdir(base_path)
    prefix = video_config["resolution"] + "_" + video_config["clip_size"] + "_" + video_config["clip_duration"] + "_" + video_name
    video_files = [
        file
        for file in files
        if (file.startswith(prefix))
    ]

    # sort the video files by file name
    video_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return video_files

def get_files_by_title(base_path, title, video_config):
    """Get video files matching the hashed title.
    Files can be .mp4, .mp3, .srt, .txt, depending on the base_path type.

    Args:
        base_path (str): Directory path to search
        title (str): Title to hash and search for
        video_config (dict): Video configuration settings

    Returns:
        tuple: (title_hash, list of matching video files)
    """
    # calculate the md5 hash of the title cut -c1-8
    command = f'echo "{title}" | md5sum | cut -c1-8'
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    title_hash = result.stdout.strip()

    # print(f"Original title: {title}, hashed title: {title_hash}")
    # get the video files that have the same md5 hash
    return title_hash, get_files_by_name(base_path, title_hash, video_config)

def generate_test_file_name(sample, task):
    """Generate a test file name from sample and task.

    Args:
        sample (str): Sample name or path
        task (str): Task identifier

    Returns:
        str: Generated test file name in format 'YYYYMMDD_sample_task'
    """
    if sample.endswith(".mp4"):
        sample = sample.split("/")[-1].split(".")[0]
    date = datetime.now().strftime("%Y%m%d")
    return f"{date}_{sample}_{task}"


# audio and video processing

def generate_audio_files(video, video_config, base_path_video, base_path_audio):
    """Extract audio from video files and save as MP3.

    Args:
        video (str): Video identifier
        video_config (dict): Video configuration settings. Should be a JSON object with "resolution" and "clip_size" keys.
        base_path_video (str): Directory containing video files
        base_path_audio (str): Directory to save audio files
    """
    video_files = get_files_by_name(base_path_video, video, video_config)
    for video_file in video_files:
        input_path = os.path.join(base_path_video, video_file)
        output_path = os.path.join(base_path_audio, video_file.replace(".mp4", ".mp3"))

        # Load video file
        video_clip = VideoFileClip(input_path)

        # Extract audio
        audio_clip = video_clip.audio

        # Save audio file
        audio_clip.write_audiofile(output_path)

        # Close resources
        audio_clip.close()
        video_clip.close()

        print(f"Audio successfully extracted and saved as: {output_path}")

def generate_transcripts(video, video_config, base_path):
    """Generate transcripts from audio files using Whisper model.

    Args:
        video (str): Video identifier
        video_config (dict): Video configuration settings. Should be a JSON object with "resolution" and "clip_size" keys.
        base_path (str): Directory containing audio files
    """
    audio_files = get_files_by_name(base_path, video, video_config)
    audio_paths = [os.path.join(base_path, audio_file) for audio_file in audio_files]
    transcripts = parallel_get_whisper("whisper", audio_paths)

    # save transcripts

    for i, transcript in enumerate(transcripts):
        with open(f"../data/transcripts/{audio_files[i].split('.')[0]}.txt", "w", encoding="utf-8") as f:
            f.write(transcript)

# load subtitles from .srt file and filter out irrelevant lines
def load_subtitle(subtitle_path):
    """Load and parse subtitle file, extracting only dialogue lines.

    Args:
        subtitle_path (str): Path to .srt subtitle file

    Returns:
        str: Concatenated dialogue lines from subtitle file
    """
    with open(subtitle_path, "r") as f:
        lines = f.readlines()
    # only keep the 2, 6, 10, ... lines
    lines = [line.strip("\n") for i, line in enumerate(lines) if i % 4 == 2]
    return " ".join(lines)

def load_transcript(transcript_path):
    """Load transcript text from file.

    Args:
        transcript_path (str): Path to transcript file

    Returns:
        str: Content of transcript file
    """
    with open(transcript_path, "r") as f:
        transcript = f.read()
    return transcript

# other utils
def refine_json_str(invalid_json):
    """Clean and format JSON string by removing markdown code blocks.

    Args:
        json_str (str): Raw JSON string with potential markdown formatting

    Returns:
        str: Cleaned JSON string
    """
    # Remove ```json or ``` from start/end
    invalid_json = invalid_json.strip()
    if invalid_json.startswith("```json"):
        invalid_json = invalid_json[7:].strip()
    if invalid_json.endswith("```"):
        invalid_json = invalid_json[:-3].strip()

    # Replace single quotes with double quotes (if needed)
    fixed_json = re.sub(r"'", '"', invalid_json)
    
    # Fix keys without double quotes
    fixed_json = re.sub(r'(?<=\{|,)\s*([a-zA-Z0-9_]+)\s*:', r'"\1":', fixed_json)
    
    # Auto-complete missing braces and brackets
    stack = []
    for char in fixed_json:
        if char in '{[':
            stack.append(char)
        elif char in '}]':
            if stack and ((char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '[')):
                stack.pop()
    
    # Complete missing brackets
    while stack:
        last = stack.pop()
        if last == '{':
            fixed_json += '}'
        elif last == '[':
            fixed_json += ']'

    # Check if quotes are balanced
    if fixed_json.count('"') % 2 != 0:
        fixed_json += '"'

    return fixed_json

def validate_and_fix_json(invalid_json):
    fixed_json = refine_json_str(invalid_json)
    try:
        # Try to parse the fixed JSON
        return json.loads(fixed_json)
    except json.JSONDecodeError as e:
        print(f"Still unable to fix: {e}")
        return None