import json
import openai
from concurrent.futures import ThreadPoolExecutor
from time import sleep
import logging
import httpx

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
    response = client[model].embeddings.create(input=text, model=model)
    return response.data[0].embedding, response.usage.total_tokens


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
    
    httpx_logger = logging.getLogger("httpx")
    original_log_level = httpx_logger.level
    httpx_logger.setLevel(logging.CRITICAL)


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda x: get_embedding_with_retry(model, x), texts))

    httpx_logger.setLevel(original_log_level)
    
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
        type can be:
            "text" - text content
            "image/jpeg", "image/png" - base64 encoded images
            "video/mp4", "video/webm" - base64 encoded videos
            "video_url" - video URL
            "audio/mp3", "audio/wav" - base64 encoded audio
        content should be a string for text,
        a list of base64 encoded media for images/video/audio,
        or a string (url) for video_url

    Returns:
        list: Formatted messages for chat completion
    """
    messages = []
    messages.append(
        {"role": "system", "content": "You are an expert in video understanding."}
    )
    content = []
    for input in inputs:
        if not input["content"]:
            print("empty content, skip")
            continue
        if input["type"] == "text":
            content.append({"type": "text", "text": input["content"]})
        elif input["type"] in ["images/jpeg", "images/png"]:
            img_format = input["type"].split("/")[1]
            if isinstance(input["content"][0], str):
                content.extend(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_format};base64,{img}",
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
                            "url": f"data:image/{img_format};base64,{img[1]}",
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
        elif input["type"] in ["video_base64/mp4", "video_base64/webm"]:
            video_format = input["type"].split("/")[1]
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:video/{video_format};base64,{input['content']}"},
                }
            )
        elif input["type"] in ["audio_base64/mp3", "audio_base64/wav"]:
            audio_format = input["type"].split("/")[1]
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:audio/{audio_format};base64,{input['content']}"
                    },
                }
            )
        else:
            raise ValueError(f"Invalid input type: {input['type']}")
    messages.append({"role": "user", "content": content})
    return messages