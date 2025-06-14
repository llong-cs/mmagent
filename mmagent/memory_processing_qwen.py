import base64
import json
import logging
from io import BytesIO
import re

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .utils.chat_api import parallel_get_embedding
from .utils.chat_qwen import generate_messages, get_response_with_retry
from .utils.general import validate_and_fix_python_list, validate_and_fix_json
from .prompts import prompt_generate_captions_with_ids_sft, prompt_generate_thinkings_with_ids_sft, prompt_generate_memory_with_ids_sft
from .memory_processing import parse_video_caption

processing_config = json.load(open("configs/processing_config.json"))
logging_level = processing_config["logging"]

MAX_RETRIES = processing_config["max_retries"]
# Configure logging
logger = logging.getLogger(__name__)

def generate_video_context(
    base64_frames, faces_list, voices_list, video_path=None, faces_input="face_only"
):
    face_frames = []
    face_only = []

    # Iterate through faces directly
    for char_id, faces in faces_list.items():
        if len(faces) == 0:
            continue
        face = faces[0]
        frame_id = face["frame_id"]
        frame_base64 = base64_frames[frame_id]

        # Convert base64 to PIL Image
        frame_bytes = base64.b64decode(frame_base64)
        frame_img = Image.open(BytesIO(frame_bytes))
        draw = ImageDraw.Draw(frame_img)

        # Draw current face
        bbox = face["bounding_box"]
        draw.rectangle(
            [(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=(0, 255, 0), width=4
        )

        # Convert back to base64
        buffered = BytesIO()
        frame_img.save(buffered, format="JPEG")
        frame_base64 = base64.b64encode(buffered.getvalue()).decode()
        face_frames.append((f"<face_{char_id}>:", frame_base64))
        face_only.append((f"<face_{char_id}>:", face["extra_data"]["face_base64"]))
    
    if faces_input == "face_only":
        faces_input = face_only
    elif faces_input == "face_frames":
        faces_input = face_frames
    else:
        raise ValueError(f"Invalid face input: {faces_input}")
    
    num_faces = len(faces_input)
    if num_faces == 0:
        logger.warning("No qualified faces detected")
    
    # Visualize face frames with IDs
    if logging_level == "DETAIL" and num_faces > 0:
        num_rows = (num_faces + 2) // 3  # Round up division to get number of rows needed

        _, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        axes = axes.ravel()  # Flatten axes array for easier indexing

        for i, face_pic in enumerate(faces_input):
            # Convert base64 to image array
            img_bytes = base64.b64decode(face_pic[1])
            img_array = np.array(Image.open(BytesIO(img_bytes)))

            axes[i].imshow(img_array)
            axes[i].set_title(face_pic[0])
            axes[i].axis("off")

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    voices_input = {}
    for id, voices in voices_list.items():
        if len(voices) == 0:
            continue
        voices_input[f"<voice_{id}>"] = [{
            "start_time": voice["start_time"],
            "end_time": voice["end_time"],
            "asr": voice["asr"]
        } for voice in voices]
    
    num_voices = len(voices_input)
    if num_voices == 0:
        logger.warning("No qualified voices detected")

    if logging_level == "DETAIL" and num_voices > 0:
        logger.debug(f"Diarized dialogues: {voices_input}")

    video_context = [
        {
            "type": "video_base64/mp4",
            "content": video_path,
        },
        {
            "type": "text",
            "content": "Face features:"
        },
        {
            "type": "images/jpeg",
            "content": faces_input,
        },
        {
            "type": "text",
            "content": "Voice features:"
        },
        {
            "type": "text",
            "content": json.dumps(voices_input),
        }
    ]

    return video_context

def generate_episodic_memories(video_context):
    input = [
        {
            "type": "text",
            "content": prompt_generate_captions_with_ids_sft,
        }
    ] + video_context

    messages = generate_messages(input)
    # print_messages(messages)
    model = "gemini-1.5-pro-002"
    
    episodic_memories = None
    for i in range(MAX_RETRIES):
        episodic_memories_string = get_response_with_retry(model, messages)[0]
        if not episodic_memories_string:
            episodic_memories_string = "[]"
            with open("logs/filtered_contents.txt", "a") as f:
                f.write(f"Filtered generated contents detected\n")
        logger.info(episodic_memories)
        episodic_memories = validate_and_fix_python_list(episodic_memories_string)
        if episodic_memories is not None:
            break
    if episodic_memories is None:
        episodic_memories = []
        # raise Exception("Failed to generate episodic memories")

    return episodic_memories

def generate_semantic_memories(video_context, video_description):
    """
    Generate thinking descriptions with character IDs based on video context and description.

    Args:
        video_context (list): List of context objects containing video information
        video_description (str): Description of the video content

    Returns:
        list: Response from the LLM model containing generated thinking descriptions with character IDs

    The function:
    1. Combines video context with description and prompt
    2. Generates messages for the LLM model
    3. Makes API call to Gemini model
    4. Returns the model's response with thinking descriptions
    """
    
    input = [
        {
            "type": "text",
            "content": prompt_generate_thinkings_with_ids_sft,
        },
    ] + video_context + [
        {
            "type": "text",
            "content": "Video descriptions:",
        },
        {
            "type": "text",
            "content": json.dumps(video_description),
        }
    ]
    messages = generate_messages(input)
    # print_messages(messages)
    model = "gemini-1.5-pro-002"
    semantic_memories = None
    for i in range(MAX_RETRIES):
        semantic_memories_string = get_response_with_retry(model, messages)[0]
        if not semantic_memories_string:
            semantic_memories_string = "[]"
            with open("logs/filtered_contents.txt", "a") as f:
                f.write(f"Filtered generated contents detected\n")
        logger.info(semantic_memories)
        semantic_memories = validate_and_fix_python_list(semantic_memories_string)
        if semantic_memories is not None:
            break
    if semantic_memories is None:
        semantic_memories = []
        # raise Exception("Failed to generate semantic memories")
    return semantic_memories

def generate_all_memories(video_context):
    input = [
        {
            "type": "text",
            "content": prompt_generate_memory_with_ids_sft,
        },
    ] + video_context
    
    messages = generate_messages(input)
    model = ""
    
    memories = None
    for i in range(MAX_RETRIES):
        memories_string = get_response_with_retry(model, messages)[0]
        if not memories_string:
            memories_string = "[]"
            with open("logs/filtered_contents.txt", "a") as f:
                f.write(f"Filtered generated contents detected\n")
        memories = validate_and_fix_json(memories_string)
        if memories is not None:
            break
    if memories is None:
        memories = {
            "video_descriptions": [],
            "high_level_conclusions": []
        }
        # raise Exception("Failed to generate memories")
    
    episodic_memories = memories["video_descriptions"]
    semantic_memories = memories["high_level_conclusions"]
    
    return episodic_memories, semantic_memories

def generate_memories(
    base64_video, base64_frames, faces_list, voices_list, video_path=None, generation_type="epi_then_sem"
):
    video_context = generate_video_context(
        base64_frames, faces_list, voices_list, video_path
    )
    
    # # get the last history_length texts
    # # deprecated
    # history_length = processing_config["history_length"]
    # history_nodes = []
    # for i in range(max(0, clip_id - history_length), clip_id):
    #     history_nodes.extend(video_graph.event_sequence_by_clip[i])
    # history_texts = [video_graph.nodes[node_id].metadata['contents'][0] for node_id in history_nodes]
    
    if generation_type == "epi_then_sem":
        episodic_memories = generate_episodic_memories(video_context)
        semantic_memories = generate_semantic_memories(video_context, episodic_memories)
    elif generation_type == "all_in_one":
        episodic_memories, semantic_memories = generate_all_memories(video_context)
    else:
        raise ValueError(f"Invalid generation type: {generation_type}")

    logger.info(episodic_memories, semantic_memories)

    return episodic_memories, semantic_memories

def process_memories(video_graph, memory_contents, clip_id, type='episodic'):
    def get_memory_embeddings(memory_contents):
        # calculate the embedding for each memory
        model = 'text-embedding-3-large'
        embeddings = parallel_get_embedding(model, memory_contents)[0]
        return embeddings

    def insert_memory(video_graph, memory, type='episodic'):
        # create a new text node for each memory
        new_node_id = video_graph.add_text_node(memory, clip_id, type)
        entities = parse_video_caption(video_graph, memory['contents'][0])
        for entity in entities:
            video_graph.add_edge(new_node_id, entity[1])

    def update_video_graph(video_graph, memories, type='episodic'):
        # append all episodic memories to the graph
        if type == 'episodic':
            # create a new text node for each memory
            for memory in memories:
                insert_memory(video_graph, memory, type)
        # semantic memories can be used to update the existing text nodes, or create new text nodes
        elif type == 'semantic':
            for memory in memories:
                entities = parse_video_caption(video_graph, memory['contents'][0])

                if len(entities) == 0:
                    insert_memory(video_graph, memory, type)
                    continue
                
                # update the existing text node for each memory, if needed
                positive_threshold = 0.85
                negative_threshold = 0
                
                # get all (possible) related nodes            
                node_id = entities[0][1]
                related_nodes = video_graph.get_connected_nodes(node_id, type=['semantic'])
                
                # if there is a node with similarity > positive_threshold, then update the edge weight by +1
                # if there is a node with similarity < negative_threshold, then update the edge weight by -1, and add a new text node and connect it to the existing node
                # otherwise, add a new text node and connect it to the existing node
                create_new_node = True
                
                for node_id in related_nodes:
                    # related nodes to be updated should satisfy two condtions:
                    # 1. the caption entities are a subset of the existing node entities
                    # 2. the semantic similarity between the memory and the existing node shows a positive correlation or a negative correlation
                    
                    # see if the memory entities are a subset of the existing node entities
                    related_node_entities = parse_video_caption(video_graph, video_graph.nodes[node_id].metadata['contents'][0])
                    embedding = video_graph.nodes[node_id].embeddings[0]
                    if all(entity in related_node_entities for entity in entities):
                        similarity = np.dot(memory['embeddings'][0], embedding) / (np.linalg.norm(memory['embeddings'][0]) * np.linalg.norm(embedding))
                        if similarity > positive_threshold:
                            video_graph.reinforce_node(node_id)
                            create_new_node = False
                        elif similarity < negative_threshold:
                            video_graph.weaken_node(node_id)
                            create_new_node = False
                
                if create_new_node:
                    insert_memory(video_graph, memory, type)
    
    memories_embeddings = get_memory_embeddings(memory_contents)

    memories = []
    for memory, embedding in zip(memory_contents, memories_embeddings):
        memories.append({
            'contents': [memory],
            'embeddings': [embedding]
        })

    update_video_graph(video_graph, memories, type)
    