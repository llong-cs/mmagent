import json

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniThinkerConfig, GenerationConfig
from transformers.utils import ModelOutput
from qwen_omni_utils import process_mm_info

def evaluate_sft(model, processor, data_path):
    model.eval()
    
    with open(data_path, "r") as f:
        for line in f:
            messages = json.loads(line)[:1]
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            try:
                USE_AUDIO_IN_VIDEO = True
                audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = inputs.to(model.device).to(model.dtype)

                # Inference: Generation of the output text and audio
                generation = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=2048)
                generate_ids = generation[:, inputs.input_ids.size(1):]
            except:
                USE_AUDIO_IN_VIDEO = False
                audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = inputs.to(model.device).to(model.dtype)

                # Inference: Generation of the output text and audio
                generation = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=2048)
                generate_ids = generation[:, inputs.input_ids.size(1):]

            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            print(response)
            print("-"*20)
            
            break

if __name__ == "__main__":
    ckpt_path = "/home/zhaoyu/workspace/Qwen/Qwen2.5-Omni-Preview/checkpoint-1000"
    val_path = "data/sft/memgen/0429/conversations/val.jsonl"
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")
    processor = Qwen2_5OmniProcessor.from_pretrained(ckpt_path)
    evaluate_sft(model, processor, val_path)
