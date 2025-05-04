import json
import os
import argparse
from tqdm import tqdm

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniThinkerConfig, GenerationConfig
from qwen_omni_utils import process_mm_info
from evaluation.memory_evaluation import eval_vdcscore, eval_autodq
from mmagent.utils.general import validate_and_fix_python_list

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="/mnt/hdfs/foundation/longlin.kylin/mmagent/data/memgen_sft/0429/ckpts")
parser.add_argument("--val_path", type=str, default="data/sft/memgen/0429/conversations/val.jsonl")
parser.add_argument("--output_dir", type=str, default="data/sft/memgen/0429/val_gen")
parser.add_argument("--val_num", type=int, default=5)
parser.add_argument("--cuda_id", type=int, default=0)
parser.add_argument("--node_num", type=int, default=8)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--generate", action="store_true")
args = parser.parse_args()


def evaluate_sft_compare(model, processor, data_path, val_num=5):
    model.eval()
    count = 0

    with open(data_path, "r") as f:
        for line in f:
            sample = json.loads(line)
            messages = sample[:1]
            gt = sample[1]["content"][0]["text"]
            if "<speaker_" in gt:
                continue
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            generation_config = GenerationConfig(pad_token_id=151643, bos_token_id=151644, eos_token_id=151645)
            try:
                USE_AUDIO_IN_VIDEO = True
                audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = inputs.to(model.device).to(model.dtype)

                # Inference: Generation of the output text and audio
                generation = model.generate(**inputs, generation_config=generation_config, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=2048)
                generate_ids = generation[:, inputs.input_ids.size(1):]
            except:
                USE_AUDIO_IN_VIDEO = False
                audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = inputs.to(model.device).to(model.dtype)

                # Inference: Generation of the output text and audio
                generation = model.generate(**inputs, generation_config=generation_config, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=2048)
                generate_ids = generation[:, inputs.input_ids.size(1):]

            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            print("GT:", gt)
            print("-"*20)
            print("SFT: ", response)

            count += 1
            if count >= val_num:
                break

def generate_sft_data(model, processor, data_path, output_dir):
    model.eval()
    samples = []
    node_num = args.node_num
    cuda_id = args.cuda_id

    with open(data_path, "r") as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)

    for i, sample in enumerate(tqdm(samples)):
        if i % node_num != cuda_id:
            continue
        messages = sample[:1]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        generation_config = GenerationConfig(pad_token_id=151643, bos_token_id=151644, eos_token_id=151645)
        try:
            USE_AUDIO_IN_VIDEO = True
            audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = inputs.to(model.device).to(model.dtype)

            # Inference: Generation of the output text and audio
            generation = model.generate(**inputs, generation_config=generation_config, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=2048)
            generate_ids = generation[:, inputs.input_ids.size(1):]
        except:
            USE_AUDIO_IN_VIDEO = False
            audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = inputs.to(model.device).to(model.dtype)

            # Inference: Generation of the output text and audio
            generation = model.generate(**inputs, generation_config=generation_config, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=2048)
            generate_ids = generation[:, inputs.input_ids.size(1):]

        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        
        with open(os.path.join(output_dir, f"{i}.json"), "w") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)

def evaluate_sft(gt_path, output_dir, val_num=10):
    gt_samples = []
    pred_samples = []
    idx = 0
    with open(gt_path, "r") as f:
        for line in f:
            sample = json.loads(line)
            res = sample[1]["content"][0]["text"]
            if "<speaker_" in res:
                idx += 1
                continue
            gt_samples.append(validate_and_fix_python_list(res))
            pred_sample = json.load(open(os.path.join(output_dir, f"{idx}.json")))
            pred_samples.append(validate_and_fix_python_list(pred_sample[1]["content"][0]["text"]))
            idx += 1
    
    assert len(gt_samples) == len(pred_samples)

    gt_samples = gt_samples[:val_num]
    pred_samples = pred_samples[:val_num]
    
    print("Evaluating AutoDQ...")
    precision, recall, f1 = eval_autodq(gt_samples, pred_samples)
    print("AutoDQ Evaluation:")
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    
    print("Evaluating VDCScore...")
    eval_dir = "data/sft/memgen/0429/evaluation"
    precision, avg_score = eval_vdcscore(gt_samples, pred_samples, os.path.join(eval_dir, f"vdcscore_evaluation_val_{val_num}.json"))
    print(f"VDCScore Evaluation:")
    print(f"Precision: {precision}, Avg Score: {avg_score}")
            
            
if __name__ == "__main__":
    ckpt_path = args.ckpt_path
    val_path = args.val_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if args.debug:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")
        processor = Qwen2_5OmniProcessor.from_pretrained(ckpt_path)
        evaluate_sft_compare(model, processor, val_path, val_num=args.val_num)
        exit()
    if args.generate:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")
        processor = Qwen2_5OmniProcessor.from_pretrained(ckpt_path)
        generate_sft_data(model, processor, val_path, output_dir)
    else:
        evaluate_sft(val_path, output_dir, val_num=args.val_num)
    
