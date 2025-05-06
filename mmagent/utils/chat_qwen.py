import json
from concurrent.futures import ThreadPoolExecutor
from time import sleep
import logging
import torch
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniThinkerConfig, GenerationConfig
from transformers.utils import ModelOutput
from qwen_omni_utils import process_mm_info

# Configure logging
logger = logging.getLogger(__name__)

processing_config = json.load(open("configs/processing_config.json"))
temp = processing_config["temperature"]
MAX_RETRIES = processing_config["max_retries"]

if torch.cuda.is_available():
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        processing_config["ckpt"],
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    thinker.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(processing_config["ckpt"])

def get_response(model, messages, timeout=30):
    """Get chat completion response from specified model.

    Args:
        model (str): Model identifier
        messages (list): List of message dictionaries

    Returns:
        tuple: (response content, total tokens used)
    """
    global thinker
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    generation_config = GenerationConfig(pad_token_id=151643, bos_token_id=151644, eos_token_id=151645)
    
    try:
        USE_AUDIO_IN_VIDEO = True
        audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(thinker.device).to(thinker.dtype)

        # Inference: Generation of the output text and audio
        with torch.no_grad():
            generation = thinker.generate(**inputs, generation_config=generation_config, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=4096, do_sample=True, temperature=temp)
            generate_ids = generation[:, inputs.input_ids.size(1):]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            token_count = len(generation[0])
            
        # Clean up
        del generation
        del generate_ids
        del inputs
        torch.cuda.empty_cache()
        
        return response, token_count
        
    except Exception as e:
        logger.warning(f"First attempt failed with audio in video: {e}")
        try:
            USE_AUDIO_IN_VIDEO = False
            audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = inputs.to(thinker.device).to(thinker.dtype)

            # Inference: Generation of the output text and audio
            with torch.no_grad():
                generation = thinker.generate(**inputs, generation_config=generation_config, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=4096, do_sample=True, temperature=temp)
                generate_ids = generation[:, inputs.input_ids.size(1):]
                response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                token_count = len(generation[0])
                
            # Clean up
            del generation
            del generate_ids
            del inputs
            torch.cuda.empty_cache()
            
            return response, token_count
            
        except Exception as e:
            logger.error(f"Second attempt failed without audio in video: {e}")
            raise

def get_response_with_retry(model, messages, timeout=30):
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
            return get_response(model, messages, timeout)
        except Exception as e:
            sleep(5)
            logger.warning(f"Retry {i} times, exception: {e}")
            continue
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")

def parallel_get_response(model, messages, timeout=30):
    """Process multiple messages in parallel using ThreadPoolExecutor.
    Messages are processed in batches, with each batch completing before starting the next.

    Args:
        model (str): Model identifier
        messages (list): List of message lists to process

    Returns:
        tuple: (list of responses, total tokens used)
    """
    batch_size = 2
    responses = []
    total_tokens = 0

    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            batch_responses = list(executor.map(lambda msg: get_response_with_retry(model, msg, timeout), batch))
            
        # Extract answers and tokens from batch responses
        batch_answers = [response[0] for response in batch_responses]
        batch_tokens = [response[1] for response in batch_responses]
        
        responses.extend(batch_answers)
        total_tokens += sum(batch_tokens)

    return responses, total_tokens

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
        inputs are like: 
        [
            {
                "type": "video_base64/mp4",
                "content": <base64>
            },
            {
                "type": "text",
                "content": "Describe the video content."
            },
            ...
        ]

    Returns:
        list: Formatted messages for chat completion
    """
    messages = []
    content = []
    for input in inputs:
        if not input["content"]:
            logger.warning("empty content, skip")
            continue
        if input["type"] == "text":
            content.append({"type": "text", "text": input["content"]})
        elif input["type"] in ["images/jpeg", "images/png"]:
            img_format = input["type"].split("/")[1]
            if isinstance(input["content"][0], str):
                content.extend(
                    [
                        {
                            "type": "image",
                            "image": f"data:image;base64,{img}",
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
                        "type": "image",
                        "image": f"data:image;base64,{img[1]}"
                    })
        elif input["type"] in ["video_url", "video_base64/mp4", "video_base64/webm"]:
            content.append(
                {
                    "type": "video",
                    "video": input["content"],
                }
            )
        else:
            raise ValueError(f"Invalid input type: {input['type']}")
    messages.append({"role": "user", "content": content})
    return messages