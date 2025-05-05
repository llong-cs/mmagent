from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniThinkerConfig, GenerationConfig
from transformers.utils import ModelOutput
from qwen_omni_utils import process_mm_info
import json
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
torch.set_printoptions(threshold=np.inf)
import sys
import os
from tqdm import tqdm

input_data = sys.argv[1] # .jsonl
output_data = sys.argv[2] # folder
index = int(sys.argv[3])

INPUT_EMBEDS = None
INPUT_MASKS = None
POSITION_IDS = None

@dataclass
class Qwen2_5OmniThinkerCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_mask: Optional[torch.FloatTensor] = None
    rope_deltas: Optional[torch.LongTensor] = None

class Qwen2_5OmniPreprocessor(Qwen2_5OmniThinkerForConditionalGeneration):

    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0

            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)
            embeds_to_talker = inputs_embeds.clone()

            # 2. Merge text , audios , image and video
            if input_ids.shape[1] != 1:
                if input_features is not None:
                    audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                        audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                    )
                    feature_lens = (
                        audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                    )
                    audio_outputs = self.audio_tower(
                        input_features,
                        feature_lens=feature_lens,
                        aftercnn_lens=audio_feat_lengths,
                    )
                    audio_features = audio_outputs.last_hidden_state
                    if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
                        raise ValueError("length of audio_features should match audio_output_lengths")
                    audio_mask = (input_ids == self.config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
                    embeds_to_talker = embeds_to_talker.masked_scatter(audio_mask, torch.zeros_like(audio_features))

                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.visual.get_dtype())
                    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                    image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                    embeds_to_talker = embeds_to_talker.masked_scatter(image_mask, torch.zeros_like(image_embeds))

                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                    video_mask = (input_ids == self.config.video_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                    embeds_to_talker = embeds_to_talker.masked_scatter(video_mask, torch.zeros_like(video_embeds))

                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)

        global INPUT_EMBEDS, INPUT_MASKS, POSITION_IDS
        INPUT_EMBEDS = inputs_embeds
        INPUT_MASKS = attention_mask
        POSITION_IDS = position_ids

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits.float()
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + ((embeds_to_talker, outputs[0])) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(embeds_to_talker, outputs.hidden_states),
            attentions=outputs.attentions,
            attention_mask=attention_mask,
            rope_deltas=rope_deltas,
        )

def find_tensor_start(A, B):
    len_A = len(A)
    len_B = len(B)
    for i in range(len_B - len_A + 1):
        if torch.all(B[i:i + len_A] == A):
            return i + len_A
    return -1

model_path = "/mnt/hdfs/foundation/agent/heyc/ckpts/Qwen2.5-Omni-7B-thinker"
model = Qwen2_5OmniPreprocessor.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

conversations = []
with open(input_data) as f:
    for line in f.readlines():
        conversations.append(json.loads(line)["messages"])
if len(conversations) % 2 != 0:
    conversations.append(conversations[0])

inputs_list = []
# It must be in batch form, otherwise it will return embeddings one by one
for idx in tqdm(range(0, len(conversations), 2)):
    if (idx // 2) % 16 != index:
        continue
    add_generation_prompt = False
    text = processor.apply_chat_template(conversations[idx: idx + 2], add_generation_prompt=add_generation_prompt, tokenize=False)
    audios, images, videos = process_mm_info(conversations[idx: idx + 2], use_audio_in_video=True)
    inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)
    generation_config = GenerationConfig(pad_token_id=151643, bos_token_id=151644, eos_token_id=151645)
    text_ids = model.generate(**inputs, generation_config=generation_config, use_audio_in_video=True, max_new_tokens=1)

    for i in range(INPUT_EMBEDS.shape[0]):

        ###################
        # mask out the trainable tokens !!!
        label_mask, mask = [], True
        for j in range(INPUT_EMBEDS.shape[1]):
            if inputs["input_ids"][i][j] == 13708 and inputs["input_ids"][i][j + 1] == 766: # <th ink / assitant
                mask = False
                label_mask.append(mask)
            elif inputs["input_ids"][i][j] == 151644: # <|im_start|> / eos
                label_mask.append(mask)
                mask = True
            else:
                label_mask.append(mask)
        label_mask[-1] = label_mask[-2] = False # <|im_end|> \n
        ###################
        
        mask = torch.tensor(label_mask)
        inputs["input_ids"][i][mask] = -100
        input_id = inputs["input_ids"][i][INPUT_MASKS[i].shape[0] - torch.sum(INPUT_MASKS[i]):]
        final_input_embed = INPUT_EMBEDS[i][INPUT_MASKS[i].shape[0] - torch.sum(INPUT_MASKS[i]):]
        position_id = POSITION_IDS[:, i, INPUT_MASKS[i].shape[0] - torch.sum(INPUT_MASKS[i]):]
        assert input_id.shape[0] == final_input_embed.shape[0]
        data_num = idx + i
        if input_id.shape[0] > 29000:
            continue
        torch.save({
            "inputs_embeds": final_input_embed.cpu(),
            "position_id": position_id.cpu(),
            "labels": input_id.cpu()
        }, os.path.join(output_data, f"{data_num}.pt"))