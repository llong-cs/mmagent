from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import os
import logging

import euler
euler.install_thrift_import_hook()
from .idl.base_thrift import *
from .idl.face_processing_thrift import *

from mmagent.utils.video_processing import process_video_clip


# Build client
# test_client = euler.Client(FaceService, 'tcp://127.0.0.1:8910', timeout=300, transport='buffered')
test_client = euler.Client(
    FaceService,
    "sd://lab.agent.face_processing_test?idc=maliva&cluster=default",
    timeout=1200,
    transport="buffered",
)

processing_config = json.load(open("configs/processing_config.json"))

cluster_size = processing_config["cluster_size"]
logger = logging.getLogger(__name__)

def process_faces(video_graph, base64_frames, save_path, preprocessing=[]):
    """
    Process video frames to detect, cluster and track faces.

    Args:
        video_graph: Graph object to store face embeddings and relationships
        base64_frames (list): List of base64 encoded video frames to process

    Returns:
        dict: Mapping of face IDs to lists of face detections, where each face detection contains:
            - frame_id (int): Frame number where face was detected
            - bounding_box (list): Face bounding box coordinates [x1,y1,x2,y2]
            - face_emb (list): Face embedding vector
            - cluster_id (int): ID of face cluster from initial clustering
            - extra_data (dict): Additional face detection metadata
            - matched_node (int): ID of matched face node in video graph

    The function:
    1. Splits frames into batches and processes them in parallel to detect faces
    2. Clusters detected faces to group similar faces together
    3. Converts face detections to JSON format
    4. Updates video graph with face embeddings and relationships
    5. Returns mapping of face IDs to face detections
    """
    batch_size = max(len(base64_frames) // cluster_size, 4)
    
    def _process_batch(params):
        """
        Process a batch of video frames to detect faces.

        Args:
            params (tuple): A tuple containing:
                - frames (list): List of video frames to process
                - offset (int): Frame offset to add to detected face frame IDs

        Returns:
            list: List of detected faces with adjusted frame IDs

        The function:
        1. Extracts frames and offset from input params
        2. Creates face detection request for the batch
        3. Gets face detection response from service
        4. Adjusts frame IDs of detected faces by adding offset
        5. Returns list of detected faces
        """
        frames = params[0]
        offset = params[1]
        req = SingleGetFaceRequest(frames=frames, Base=Base())
        resp = test_client.SingleGetFace(req)
        faces = resp.faces
        for face in faces:
            face.frame_id += offset
        return faces

    def get_embeddings(base64_frames, batch_size):
        num_batches = (len(base64_frames) + batch_size - 1) // batch_size
        batched_frames = [
            (base64_frames[i * batch_size : (i + 1) * batch_size], i * batch_size)
            for i in range(num_batches)
        ]

        faces = []

        # parallel process the batches
        with ThreadPoolExecutor(max_workers=num_batches) as executor:
            for batch_faces in tqdm(
                executor.map(_process_batch, batched_frames), total=num_batches
            ):
                faces.extend(batch_faces)

        req = SingleClusterFaceRequest(faces=faces, Base=Base())
        resp = test_client.SingleClusterFace(req)

        faces = resp.faces

        return faces

    def establish_mapping(faces, key="cluster_id", filter=None):
        mapping = {}
        for face in faces:
            if key not in face.keys():
                raise ValueError(f"key {key} not found in faces")
            if filter and not filter(face):
                continue
            id = face[key]
            if id not in mapping:
                mapping[id] = []
            mapping[id].append(face)
        # sort the faces in each cluster by detection score and quality score
        max_faces = processing_config["max_faces_per_character"]
        for id in mapping:
            mapping[id] = sorted(
                mapping[id],
                key=lambda x: (
                    float(x["extra_data"]["face_detection_score"]),
                    float(x["extra_data"]["face_quality_score"]),
                ),
                reverse=True,
            )[:max_faces]
        return mapping

    def filter_score_based(face):
        dthresh = processing_config["face_detection_score_threshold"]
        qthresh = processing_config["face_quality_score_threshold"]
        return float(face["extra_data"]["face_detection_score"]) > dthresh and float(face["extra_data"]["face_quality_score"]) > qthresh

    def update_videograph(video_graph, tempid2faces):
        id2faces = {}
        for tempid, faces in tempid2faces.items():
            if tempid == -1:
                continue
            if len(faces) == 0:
                continue
            face_info = {
                "embeddings": [face["face_emb"] for face in faces],
                "contents": [face["extra_data"]["face_base64"] for face in faces],
            }
            matched_nodes = video_graph.search_img_nodes(face_info)
            if len(matched_nodes) > 0:
                matched_node = matched_nodes[0][0]
                video_graph.update_node(matched_node, face_info)
                for face in faces:
                    face["matched_node"] = matched_node
            else:
                matched_node = video_graph.add_img_node(face_info)
                for face in faces:
                    face["matched_node"] = matched_node
            if matched_node not in id2faces:
                id2faces[matched_node] = []
            id2faces[matched_node].extend(faces)
        
        max_faces = processing_config["max_faces_per_character"]
        for id, faces in id2faces.items():
            id2faces[id] = sorted(
                faces,
                key=lambda x: (
                    float(x["extra_data"]["face_detection_score"]),
                    float(x["extra_data"]["face_quality_score"]),
                ),
                reverse=True
            )[:max_faces]

        return id2faces
    
    # Check if intermediate results exist
    try:
        with open(save_path, "r") as f:
            faces_json = json.load(f)
    except Exception as e:
        try:
            faces = get_embeddings(base64_frames, batch_size)

            faces_json = [
                {
                    "frame_id": face.frame_id,
                    "bounding_box": face.bounding_box,
                    "face_emb": face.face_emb,
                    "cluster_id": face.cluster_id,
                    "extra_data": face.extra_data,
                }
                for face in faces
            ]

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
            with open(save_path, "w") as f:
                json.dump(faces_json, f)
            
            logger.info(f"Write face detection results to {save_path}")
        except Exception as e:
            # Save error to log file
            log_dir = processing_config["log_dir"]
            os.makedirs(log_dir, exist_ok=True)
            error_log_path = os.path.join(log_dir, "error_face_preprocessing.log")
            with open(error_log_path, "a") as f:
                f.write(f"Error processing {save_path}: {str(e)}\n")
            logger.error(f"Failed to detect faces at {save_path}: {e}")
            raise RuntimeError(f"Failed to detect faces at {save_path}: {e}")
            
    if "face" in preprocessing:
        return

    if len(faces_json) == 0:
        return {}

    tempid2faces = establish_mapping(faces_json, key="cluster_id", filter=filter_score_based)
    if len(tempid2faces) == 0:
        return {}

    id2faces = update_videograph(video_graph, tempid2faces)

    return id2faces

def main():
    _, frames, _ = process_video_clip(
        "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/video_clips/CZ_2/-OCrS_r5GHc/11.mp4"
    )
    process_faces(None, frames, "data/temp/face_detection_results.json", preprocessing=["face"])

if __name__ == "__main__":
    main()