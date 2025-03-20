import base64
import json
import ast
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from io import BytesIO

from utils.chat_api import *
from prompts import *



def generate_thinkings_with_ids(video_context, video_description):
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
    response = get_response_with_retry(model, messages)

    return response


def generate_captions_and_thinkings_with_ids(
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

    # print(video_url)
    print(len(base64_video))
    video_context = [
        {
            "type": "video_base64/mp4",
            "content": base64_video,
        },
        {
            "type": "images/jpeg",
            "content": face_frames,
        },
        {
            "type": "text",
            "content": json.dumps(voices_list),
        },
    ]
    input = video_context + [
        {
            "type": "text",
            "content": prompt_generate_captions_with_ids,
        }
    ]

    messages = generate_messages(input)
    model = "gemini-1.5-pro-002"
    captions = get_response_with_retry(model, messages)

    # Visualize face frames with IDs
    num_faces = len(face_frames)
    num_rows = (num_faces + 2) // 3  # Round up division to get number of rows needed

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
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

    print(voices_list)

    thinkings = generate_thinkings_with_ids(video_context, captions[0])

    return captions[0], thinkings[0]

def process_descriptions(video_graph, video_descriptions_string):
    def string_to_list(s):
        try:
            # Remove ```json or ``` from start/end
            s = s.strip("```json").strip("```")
            result = ast.literal_eval(s)
            if isinstance(result, list):
                return result
            else:
                raise ValueError("Input string is not a list")
        except (SyntaxError, ValueError) as e:
            print(f"Parsing error: {e}")
            return None

    def parse_video_description(video_description):
        # video_description is a string like this: <char_1> xxx <char_2> xxx
        # extract all the elements wrapped by < and >
        entities = []
        current_entity = ""
        in_entity = False

        for char in video_description:
            if char == "<":
                in_entity = True
                current_entity = ""
            elif char == ">":
                in_entity = False
                node_type, node_id = current_entity.split("_")
                # TODO: check node_id dtype
                entities.append((node_type, node_id))
            else:
                if in_entity:
                    current_entity += char
        return entities

    def update_video_graph(video_graph, descriptions):
        for description in descriptions:
            new_node_id = video_graph.add_text_node(description)
            entities = parse_video_description(description)
            for _, node_id in entities:
                video_graph.add_edge(new_node_id, node_id)

    descriptions = string_to_list(video_descriptions_string)
    print(descriptions)
    update_video_graph(video_graph, descriptions)