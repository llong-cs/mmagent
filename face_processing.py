import euler
euler.install_thrift_import_hook()
from idl.base_thrift import *
from idl.face_processing_thrift import *

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Build client
# test_client = euler.Client(FaceService, 'tcp://127.0.0.1:8910', timeout=300, transport='buffered')
test_client = euler.Client(
    FaceService,
    "sd://lab.agent.face_processing_test?idc=maliva&cluster=default",
    timeout=300,
    transport="buffered",
)

CLUSTER_SIZE = 100

def process_batch(params):
    frames = params[0]
    offset = params[1]
    req = SingleGetFaceRequest(frames=frames, Base=Base())
    resp = test_client.SingleGetFace(req)
    faces = resp.faces
    for face in faces:
        face.frame_id += offset
    return faces


def process_faces(video_graph, base64_frames):
    batch_size = len(base64_frames) // CLUSTER_SIZE
    
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
                executor.map(process_batch, batched_frames), total=num_batches
            ):
                faces.extend(batch_faces)

        req = SingleClusterFaceRequest(faces=faces, Base=Base())
        resp = test_client.SingleClusterFace(req)

        faces = resp.faces

        return faces

    def establish_mapping(faces, key="cluster_id"):
        mapping = {}
        if key in faces[0].keys():
            for face in faces:
                id = face[key]
                if id not in mapping:
                    mapping[id] = []
                mapping[id].append(face)
        else:
            raise ValueError(f"key {key} not found in faces")
        # sort the faces in each cluster by detection score and quality score
        for id in mapping:
            mapping[id] = sorted(
                mapping[id],
                key=lambda x: (
                    float(x["extra_data"]["face_detection_score"]),
                    float(x["extra_data"]["face_quality_score"]),
                ),
                reverse=True,
            )
        return mapping

    def filter_score_based(faces):
        dthresh = 0.85
        qthresh = 22
        max_faces = 3
        filtered_faces = [
            face
            for face in faces
            if float(face["extra_data"]["face_detection_score"]) > dthresh
            and float(face["extra_data"]["face_quality_score"]) > qthresh
        ]
        return filtered_faces[:max_faces]

    def update_videograph(video_graph, tempid2faces, filter=None):
        faces_list = []
        for tempid, faces in tempid2faces.items():
            if tempid == -1:
                continue
            if filter:
                filtered_faces = filter(faces)
            else:
                filtered_faces = faces
            face_embs = [face["face_emb"] for face in filtered_faces]
            matched_nodes = video_graph.search_img_nodes(face_embs)
            if len(matched_nodes) > 0:
                matched_node = matched_nodes[0][0]
                video_graph.add_embedding(matched_node, face_embs)
                for face in faces:
                    face["matched_node"] = matched_node
            else:
                matched_node = video_graph.add_img_node(face_embs)
                for face in faces:
                    face["matched_node"] = matched_node
            faces_list.extend(filtered_faces)

        return faces_list

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

    tempid2faces = establish_mapping(faces_json, key="cluster_id")

    tagged_faces_json = update_videograph(
        video_graph, tempid2faces, filter=filter_score_based
    )

    id2faces = establish_mapping(tagged_faces_json, key="matched_node")

    return id2faces