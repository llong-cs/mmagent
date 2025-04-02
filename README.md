# Memory-Enhanced Multimodal Agent

## Preparation

### Environment

Make sure your python version is **below 3.10**.

```bash
pip install -r requirements.txt
```

### Data

Set `video_paths` in `configs/processing_config.json`.

Video path can be

1. **path to a .mp4 file**, or
2. **path to a directory** that contains preprocessed video clips.

For video preprocessing, you can follow `utils/video_processing.py` for end-to-end video segmentation.

### Configuration

`configs/processing_config.json`

```json
{
    "video_paths": [
        "data/videos/clipped/5 Poor People vs 1 Secret Millionaire"
    ], // videos to be processed
    "interval_seconds": 30, // clip length (in seconds)
    "fps": 5, // frame rate used for face detection
    "segment_limit": -1, // number of clips to be processed (e.g., 3 means processing the first 3 clips of the video, -1 means processing the entire video)
    "cluster_size": 100, // scale of face detection service
    "face_detection_score_threshold": 0.85,
    "face_quality_score_threshold": 22, // a face is considered qualified if its quality score >= face_quality_score_threshold and its quality score >= face_quality_score_threshold
    "max_faces_per_character": 3, // number of faces used to define an individual detected in the clip
    "min_duration_for_audio": 2, // a voice segmentation is considered qualified if its duration >= min_duration_for_audio
    "history_length": 1, // when generating episodic and semantic memories, the model can see episodic memories generated from the last history_length clips
    "max_retries": 3,
    "query_num": 10, // number of queries generated per retrieval
    "topk": 5, // (maximum) number of clips can be retrieved by each query
    "max_retrieval_steps": 3, // maximum number of retrieval steps
    "logging": "INFO",
    "save_dir": "data/mems", // saving directory of the generated memory graphs
    "intermediate_save_dir": "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/intermediate_outputs", // saving directory of the intermediate outputs
    "input_dir": "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/video_clips", 
    "max_parallel_videos": 16,
    "log_dir": "logs"
}
```

`configs/memory_config.json`

```json
{
    "max_img_embeddings": 10, // maximum number of faces that can be stored in the memroy per character
    "max_audio_embeddings": 20, // maximum number of voices that can be stored in the memroy per character
    "img_matching_threshold": 0.3, // two faces are considered belonging to the same individual if the cosine simlarity of their embeddings >= face_detection_score_threshold
    "audio_matching_threshold": 0.6 // two voices are considered belonging to the same individual if the cosine simlarity of their embeddings >= audio_matching_threshold
}
```

## Running

### Generate memory

1. Set up configs.
2. Follow `demo.ipynb` to generate memory graphs for your videos.
3. The generated memory graphs will be saved at `data/mems`.

### Answer with retrieval

1. Load a memory graph from `data/mems`.
2. Design a question.
3. Follow `demo.ipynb` to answer a question based on the loaded video graph.

### Memory visualization

1. Load a memory graph from `data/mems`.
2. Follow `demo.ipynb` to visualize the loaded video graph.