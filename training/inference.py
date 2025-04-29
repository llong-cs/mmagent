from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniThinkerConfig, GenerationConfig
from qwen_omni_utils import process_mm_info
import json
import torch

# config = Qwen2_5OmniThinkerConfig.from_pretrained("/mnt/hdfs/foundation/agent/heyc/ckpts/Qwen2.5-Omni-7B-thinker")
# model = Qwen2_5OmniThinkerForConditionalGeneration(config)
# model.load_state_dict(torch.load("/mnt/hdfs/foundation/agent/heyc/ckpts/Qwen2.5-Omni-7B-thinker/pytorch_model.bin"))

# model_path = "/mnt/hdfs/foundation/agent/heyc/ckpts/Qwen2.5-Omni-7B"
# model = Qwen2_5OmniModel.from_pretrained(
#     model_path,
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )
# model = Qwen2_5OmniThinkerForConditionalGeneration(model.config.thinker_config)


model_path = "/mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/sft_output"
# model_path = "/mnt/hdfs/foundation/agent/heyc/ckpts/Qwen2.5-Omni-7B-thinker"
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

conversations = []
with open("data/sft_data.jsonl") as f:
    for line in f.readlines():
        conversations.append(json.loads(line)["messages"][:-1])
conversations = conversations[:4]

# set use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for batch inference
text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)
for k, v in inputs.items():
    print(k, v.shape)
exit()
# print(inputs["input_ids"])
generation_config = GenerationConfig(pad_token_id=151643, bos_token_id=151644, eos_token_id=151644)
text_ids = model.generate(**inputs, generation_config=generation_config, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=256)
text = processor.batch_decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
print(text)