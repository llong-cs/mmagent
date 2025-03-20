from utils.chat_api import *
from prompts import *

import base64
from PIL import Image
from io import BytesIO
import numpy as np

# Check if the given faces are recognizable
def batch_classify_faces(faces):
    print(len(faces))
    base64_faces = [face["extra_data"]["face_base64"] for face in faces]
    inputs = [
        [
            {"type": "images/jpeg", "content": [base64_face]},
            {"type": "text", "content": prompt_classify_recognizable_faces},
        ]
        for base64_face in base64_faces
    ]
    messages = [generate_messages(input) for input in inputs]
    model = "gemini-1.5-pro-002"
    response = parallel_get_response(model, messages)
    for i in range(len(response[0])):
        faces[i]["extra_data"]["recognizable"] = int(response[0][i])
    return faces

def select_representative_faces_with_rules(faces):
    """Select the most representative face for each cluster based on face type, size and similarity.

    Args:
        faces (list): List of face dictionaries containing frame_id, bounding_box, face_emb,
                     cluster_id and extra_data with face_type

    Returns:
        dict: Mapping of cluster_id to the most representative face
    """
    # Group faces by cluster
    clusters = {}
    for face in faces:
        cluster_id = face["cluster_id"]
        if cluster_id == -1:  # Skip noise points
            continue
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(face)

    representative_faces = {}

    # For each cluster, find the best representative face
    for cluster_id, cluster_faces in clusters.items():
        # First try to find ortho faces
        ortho_faces = []
        side_faces = []
        for f in cluster_faces:
            if f["extra_data"]["face_type"] == "ortho":
                ortho_faces.append(f)
            else:
                side_faces.append(f)

        if ortho_faces:
            # For ortho faces, first select top 10% by size
            areas = [
                (
                    f,
                    (f["bounding_box"][2] - f["bounding_box"][0])
                    * (f["bounding_box"][3] - f["bounding_box"][1]),
                )
                for f in ortho_faces
            ]
            areas.sort(key=lambda x: x[1], reverse=True)
            top_size_faces = [f[0] for f in areas[: max(1, int(len(areas) * 0.1))]]

            # If only one face remains, use it directly
            if len(top_size_faces) == 1:
                best_face = top_size_faces[0]
            else:
                # Find the one with highest average similarity to all faces in cluster
                max_avg_similarity = -1
                best_face = None
                cluster_embeddings = np.array(
                    [face["face_emb"] for face in cluster_faces]
                )

                for face in top_size_faces:
                    similarities = np.dot(cluster_embeddings, face["face_emb"])
                    avg_similarity = (np.sum(similarities) - 1) / (
                        len(cluster_faces) - 1
                    )
                    if avg_similarity > max_avg_similarity:
                        max_avg_similarity = avg_similarity
                        best_face = face

        else:
            # For side faces, first select top 20% by aspect ratio closest to 1
            if side_faces:
                areas = [
                    (
                        f,
                        (f["bounding_box"][2] - f["bounding_box"][0])
                        * (f["bounding_box"][3] - f["bounding_box"][1]),
                    )
                    for f in side_faces
                ]
                areas.sort(key=lambda x: x[1], reverse=True)
                top_area_faces = [f[0] for f in areas[: max(1, int(len(areas) * 0.5))]]

                # Then select top 20% by aspect ratio closest to 1
                ratios = []
                for face in top_area_faces:
                    bbox = face["bounding_box"]
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    ratio = abs(width / height - 1.0)
                    ratios.append((face, ratio))

                ratios.sort(key=lambda x: x[1])  # Sort by ratio difference from 1
                final_candidates = [
                    f[0] for f in ratios[: max(1, int(len(ratios) * 0.2))]
                ]

                # If only one face remains, use it directly
                if len(final_candidates) == 1:
                    best_face = final_candidates[0]
                else:
                    # Find the one with highest average similarity to all faces in cluster
                    max_avg_similarity = -1
                    best_face = None
                    cluster_embeddings = np.array(
                        [face["face_emb"] for face in cluster_faces]
                    )

                    for face in final_candidates:
                        similarities = np.dot(cluster_embeddings, face["face_emb"])
                        avg_similarity = (np.sum(similarities) - 1) / (
                            len(cluster_faces) - 1
                        )
                        if avg_similarity > max_avg_similarity:
                            max_avg_similarity = avg_similarity
                            best_face = face

        representative_faces[cluster_id] = best_face

    # return representative_faces

    faces_list = []
    for cluster_id, face in representative_faces.items():
        faces_list.append(face)
    return faces_list

def select_representative_faces_with_gpt(faces):
    # Group faces by cluster
    clusters = {}
    for face in faces:
        cluster_id = face["cluster_id"]
        if cluster_id == -1:  # Skip noise points
            continue
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(face)

    faces_list = []

    # For each cluster, find the best representative face
    for cluster_id, cluster_faces in clusters.items():
        faces_base64 = [face["extra_data"]["face_base64"] for face in cluster_faces]
        print(f"faces number: {len(faces_base64)}")
        # model = 'gemini-1.5-pro-002'
        # input = [
        #     {
        #         "type": "images",
        #         "content": faces_base64,
        #     },
        #     {
        #         "type": "text",
        #         "content": prompt_select_representative_faces,
        #     }
        # ]
        model = "gpt-4o-2024-11-20"
        input = [
            {
                "type": "images/jpeg",
                "content": faces_base64,
            },
            {
                "type": "text",
                "content": prompt_select_representative_faces_forced,
            },
        ]
        messages = generate_messages(input)

        response = get_response_with_retry(model, messages)
        try:
            index = int(response[0])
            if index >= 0:
                print(f"best face: {index}")
                faces_list.append(cluster_faces[index])
            else:
                print(f"cannot find a good face")
                # insert a face with black base64
                size = (100, 100)
                black_image = Image.new("RGB", size, (0, 0, 0))
                buffered = BytesIO()
                black_image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                black_face = {
                    "frame_id": -1,
                    "bounding_box": [0, 0, 0, 0],
                    "face_emb": np.zeros_like(cluster_faces[0]["face_emb"]).tolist(),
                    "cluster_id": -1,
                    "extra_data": {"face_type": "other", "face_base64": img_base64},
                }
                faces_list.append(black_face)
        except:
            print(f"cannot find a good face")
            # insert a face with black base64
            size = (100, 100)
            black_image = Image.new("RGB", size, (0, 0, 0))
            buffered = BytesIO()
            black_image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            black_face = {
                "frame_id": -1,
                "bounding_box": [0, 0, 0, 0],
                "face_emb": np.zeros_like(cluster_faces[0]["face_emb"]).tolist(),
                "cluster_id": -1,
                "extra_data": {"face_type": "other", "face_base64": img_base64},
            }
            faces_list.append(black_face)

    return faces_list

def select_representative_faces_with_scores(faces, max_faces=3):
    # Group faces by cluster
    clusters = {}
    for face in faces:
        cluster_id = face["cluster_id"]
        if cluster_id == -1:  # Skip noise points
            continue
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(face)

    faces_list = []
    faces_per_cluster = {}
    dthresh = 0.85
    qthresh = 22

    # For each cluster, find the best representative face
    for cluster_id, cluster_faces in clusters.items():
        qualified_faces = [
            face
            for face in cluster_faces
            if float(face["extra_data"]["face_detection_score"]) > dthresh
            and float(face["extra_data"]["face_quality_score"]) > qthresh
        ]
        if qualified_faces:
            # Sort faces by face_detection_score and face_quality_score
            sorted_faces = sorted(
                qualified_faces,
                key=lambda x: (
                    float(x["extra_data"]["face_detection_score"]),
                    float(x["extra_data"]["face_quality_score"]),
                ),
                reverse=True,
            )
            # Select the face with the highest face_detection_score and face_quality_score
            best_faces = sorted_faces[:max_faces]
            faces_per_cluster[cluster_id] = best_faces
            faces_list.append(best_faces)

    return faces_list
    # return faces_per_cluster