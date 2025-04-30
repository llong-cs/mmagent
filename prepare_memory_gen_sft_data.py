import os
import json
from mmagent.prompts import prompt_generate_captions_with_ids_sft, prompt_generate_thinkings_with_ids_sft

data_path = "data/annotations/sft/memgen/0429/train_for_memory_5k.json"

def fix_equivalences_format(data_path):
    samples = []
    with open(data_path, "r") as f:
        data = json.load(f)
    for item in data:
        samples.extend(item["clips"])
    
    for sample in samples:
        with open(sample, "r") as f:
            data = json.load(f)
            
        equivalences = data["semantic_memory"]
        new_equivalences = []
        for equivalence in equivalences:
            id1, id2 = equivalence.split("is")
            id1, id2 = id1.strip(), id2.strip()
            new_equivalences.append(f"Equivalence: {id1}, {id2}")
        data["semantic_memory"] = new_equivalences
        
        with open(sample, "w") as f:
            json.dump(data, f, indent=4)

def generate_video_context(data):
    message_content = [
        {
            "type": "video",
            "video": data["video"]
        }
    ]

    message_content.append({
        "type": "text",
        "text": "Face features:"
    })
    
    for face_id, face_path in data["characters"].items():
        message_content.extend([
            {
                "type": "text",
                "text": face_id
            },
            {
                "type": "image",
                "image": face_path
            }
        ])
    
    message_content.append({
        "type": "text",
        "text": "Voice features:"
    })
    
    voice_info = {}
    for voice_id, speech_segments in data["speakers"].items():
        voice_info[voice_id] = [{
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "asr": segment["asr"]
        } for segment in speech_segments]
    
    message_content.append({
        "type": "text",
        "text": json.dumps(voice_info)
    })

    return message_content

def generate_episodic_conversations(data_path, output_path):
    samples = []
    with open(data_path, "r") as f:
        data = json.load(f)
    for item in data:
        samples.extend(item["clips"])
    
    for sample in samples:
        with open(sample, "r") as f:
            data = json.load(f)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_generate_captions_with_ids_sft
                    }
                ]
            }
        ]
        
        messages[0]["content"].extend(generate_video_context(data))
        
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(data["episodic_memory"])
                }
            ]
        })

        res = {
            "messages": messages
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(res) + "\n")

def generate_semantic_conversations(data_path, output_path, sem_mem_types=["semantic_memory", "semantic_memory_character", "semantic_memory_relation", "semantic_memory_video", "semantic_memory_general"]):
    samples = []
    with open(data_path, "r") as f:
        data = json.load(f)
    for item in data:
        samples.extend(item["clips"])
    
    for sample in samples:
        with open(sample, "r") as f:
            data = json.load(f)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_generate_thinkings_with_ids_sft
                    }
                ]
            }
        ]
        
        messages[0]["content"].extend(generate_video_context(data))
        
        messages[0]["content"].extend([
            {
                "type": "text",
                "text": "Video descriptions:"
            },
            {
                "type": "text",
                "text": json.dumps(data["episodic_memory"])
            }
        ])
        
        sem_mems = []
        for sem_mem_type in sem_mem_types:
            sem_mems.extend(data[sem_mem_type])
        
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(sem_mems)
                }
            ]
        })

        res = {
            "messages": messages
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    fix_equivalences_format(data_path)
    generate_episodic_conversations(data_path, "data/annotations/sft/memgen/0429/episodic_conversations.jsonl")
    generate_semantic_conversations(data_path, "data/annotations/sft/memgen/0429/semantic_conversations.jsonl")