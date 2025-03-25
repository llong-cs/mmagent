"""
This file contains functions for processing video descriptions and generating captions and thinkings with character IDs.
"""

import base64
import json
import ast
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from utils.chat_api import generate_messages, get_response_with_retry, parallel_get_embedding
from utils.general import validate_and_fix_python_list
from prompts import prompt_generate_captions_with_ids, prompt_generate_thinkings_with_ids

MAX_RETRIES = 3

def generate_thinkings_with_ids(video_context, video_description):
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
    input = video_context + [
        {
            "type": "text",
            "content": f"video_description: {video_description}",
        },
        {
            "type": "text",
            "content": prompt_generate_thinkings_with_ids,
        },
    ]
    messages = generate_messages(input)
    model = "gemini-1.5-pro-002"
    thinkings = None
    for i in range(MAX_RETRIES):
        print(f"Generating thinkings {i} times")
        thinkings_string = get_response_with_retry(model, messages)[0]
        thinkings = validate_and_fix_python_list(thinkings_string)
        if thinkings is not None:
            break
    if thinkings is None:
        raise Exception("Failed to generate thinkings")
    return thinkings

def generate_video_context(
    base64_video, base64_frames, base64_audio, faces_list, voices_list
):
    face_frames = []

    print(f"id num: {len(faces_list)}")
    # print(len(faces_list[0]))

    # Iterate through faces directly
    for char_id, faces in faces_list.items():
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
        face_frames.append((f"<char_{char_id}>:", frame_base64))
    
    # Visualize face frames with IDs
    num_faces = len(face_frames)
    num_rows = (num_faces + 2) // 3  # Round up division to get number of rows needed

    _, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    axes = axes.ravel()  # Flatten axes array for easier indexing

    for i, face_frame in enumerate(face_frames):
        # Convert base64 to image array
        img_bytes = base64.b64decode(face_frame[1])
        img_array = np.array(Image.open(BytesIO(img_bytes)))

        axes[i].imshow(img_array)
        axes[i].set_title(face_frame[0])
        axes[i].axis("off")

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    voices_input = {}
    for id, voices in voices_list.items():
        voices_input[f"<speaker_{id}>"] = [{
            "start_time": voice["start_time"],
            "end_time": voice["end_time"],
            "content": voice["asr"]
        } for voice in voices]

    print(voices_input)

    video_context = [
        {
            "type": "video_base64/mp4",
            "content": base64_video.decode("utf-8"),
        },
        {
            "type": "images/jpeg",
            "content": face_frames,
        },
        {
            "type": "text",
            "content": json.dumps(voices_input),
        },
    ]

    return video_context

def generate_captions_and_thinkings_with_ids(
    base64_video, base64_frames, base64_audio, faces_list, voices_list
):
    """
    Generate captions and thinking descriptions for video content with character IDs.

    Args:
        base64_video (bytes): Base64 encoded video data
        base64_frames (list): List of base64 encoded video frames
        base64_audio (bytes): Base64 encoded audio data
        faces_list (dict): Dictionary mapping character IDs to lists of face detections
        voices_list (list): List of voice/speech segments detected in the audio

    Returns:
        tuple: A tuple containing:
            - str: Generated captions with character ID references
            - str: Generated thinking descriptions with character ID references

    The function:
    1. Extracts face frames for each character and draws bounding boxes
    2. Creates a context object with video, face frames and voice data
    3. Generates captions using an LLM model
    4. Visualizes the face frames with character IDs
    5. Generates thinking descriptions based on the captions
    """
    video_context = generate_video_context(
        base64_video, base64_frames, base64_audio, faces_list, voices_list
    )

    input = video_context + [
        {
            "type": "text",
            "content": prompt_generate_captions_with_ids,
        }
    ]

    messages = generate_messages(input)
    model = "gemini-1.5-pro-002"
    captions = None
    for i in range(MAX_RETRIES):
        print(f"Generating captions {i} times")
        captions_string = get_response_with_retry(model, messages)[0]
        captions = validate_and_fix_python_list(captions_string)
        if captions is not None:
            break
    if captions is None:
        raise Exception("Failed to generate captions")

    thinkings = generate_thinkings_with_ids(video_context, captions)

    return captions, thinkings

def process_captions(video_graph, caption_contents, type='episodic'):
    """
    Process video descriptions and update the video graph with text nodes and edges.
    
    Args:
        video_graph: The video graph object to update
        video_captions_string: String containing video captions in JSON format, with entity references
            in the format <entity_type_id>. For example: "<char_1> walks to <char_2>"
            
    The function:
    1. Converts the JSON string to a list of captions
    2. For each caption:
        - Creates a new text node with the caption
        - Extracts entity references (e.g. char_1, char_2)
        - Adds edges between the text node and referenced entity nodes
    """
    def get_caption_embeddings(caption_contents, prefix='Memory: ', suffix='Based on the memory, you can learn that'):
        # calculate the embedding for each caption
        model = 'text-embedding-3-large'

        caption_contents = [prefix + caption + suffix for caption in caption_contents]

        embeddings = parallel_get_embedding(model, caption_contents)[0]
        return embeddings

    def parse_video_caption(video_caption):
        # video_caption is a string like this: <char_1> xxx <char_2> xxx
        # extract all the elements wrapped by < and >
        entities = []
        current_entity = ""
        in_entity = False

        for char in video_caption:
            if char == "<": 
                in_entity = True
                current_entity = ""
            elif char == ">":
                if in_entity:
                    in_entity = False
                    node_type, node_id = current_entity.split("_")
                    node_id = int(node_id)
                    entities.append((node_type, node_id))
            else:
                if in_entity:
                    current_entity += char
        return entities

    def insert_caption(video_graph, caption, type='episodic'):
        # create a new text node for each caption
        new_node_id = video_graph.add_text_node(caption, type)
        entities = parse_video_caption(caption['content'])
        for entity in entities:
            video_graph.add_edge(new_node_id, entity[1])

    def update_video_graph(video_graph, captions, type='episodic'):
        # append all episodic captions to the graph
        if type == 'episodic':
            # create a new text node for each caption
            for caption in captions:
                insert_caption(video_graph, caption, type)
        # semantic captions can be used to update the existing text nodes, or create new text nodes
        elif type == 'semantic':
            for caption in captions:
                entities = parse_video_caption(caption)

                if len(entities) == 0:
                    insert_caption(video_graph, caption, type)
                    continue
                
                # update the existing text node for each caption, if needed
                positive_threshold = 0.9
                negative_threshold = -0.5
                
                # get all (possible) related nodes            
                node_id = entities[0][1]
                related_nodes = video_graph.get_connected_nodes(node_id, type='semantic')
                
                # if there is a node with similarity > positive_threshold, then update the edge weight by +1
                # if there is a node with similarity < negative_threshold, then update the edge weight by -1, and add a new text node and connect it to the existing node
                # otherwise, add a new text node and connect it to the existing node
                create_new_node = True
                
                for node_id in related_nodes:
                    # related nodes to be updated should satisfy two condtions:
                    # 1. the caption entities are a subset of the existing node entities
                    # 2. the semantic similarity between the caption and the existing node shows a positive correlation or a negative correlation
                    
                    # see if the caption entities are a subset of the existing node entities
                    related_node_entities = parse_video_caption(video_graph.nodes[node_id].metadata['text'])
                    embedding = video_graph.nodes[node_id].embeddings[0]
                    if all(entity in related_node_entities for entity in entities):
                        similarity = np.dot(caption['embedding'], embedding) / (np.linalg.norm(caption['embedding']) * np.linalg.norm(embedding))
                        if similarity > positive_threshold:
                            video_graph.reinforce_node(node_id)
                            create_new_node = False
                        elif similarity < negative_threshold:
                            video_graph.weaken_node(node_id)
                            create_new_node = False
                
                if create_new_node:
                    insert_caption(video_graph, caption, type)
    
    print(caption_contents)
    
    captions_embeddings = get_caption_embeddings(caption_contents)

    captions = []
    for caption, embedding in zip(caption_contents, captions_embeddings):
        captions.append({
            'content': caption,
            'embedding': embedding
        })

    update_video_graph(video_graph, captions, type)
    