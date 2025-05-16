# Memory-Enhanced Multimodal Agent

## Preparation

### Environment

```bash
pip3 install -r requirements.txt
pip3 install -e .

pip3 install accelerate==0.34.2 # https://github.com/huggingface/trl/issues/2377
pip3 install qwen-omni-utils==0.0.4

sudo pip3 uninstall -y transformers
sudo pip3 install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
pip3 install flash-attn==2.6.3 --no-build-isolation

sudo apt-get -y install ffmpeg # load audio in video(mp4)
```

### Data

Set `video_paths` in `configs/processing_config.json`.

Video path can be

1. **path to a .mp4 file**, or
2. **path to a directory** that contains preprocessed video clips.

For video preprocessing, you can follow `mmagent/utils/video_processing.py` for end-to-end video segmentation.

### Configuration

For `configs/processing_config.json`:

- `video_paths`: List of videos to be processed.

- `interval_seconds`: Length of each clip in seconds.

- `fps`: Frame rate used for face detection.

- `segment_limit`: Number of clips to process (-1 for entire video).

- `cluster_size`: Scale of face detection service.

- `face_detection_score_threshold`: Threshold for face detection score.

- `face_quality_score_threshold`: Minimum score for a face to be considered qualified.

- `max_faces_per_character`: Number of faces used to define an individual in a clip.

- `min_duration_for_audio`: Minimum duration for a voice segment to be qualified.

- `history_length`: Number of previous clips' episodic memories visible when generating new memories.

- `max_retries`: Maximum number of retry attempts.

- `query_num`: Number of queries generated per retrieval.

- `topk`: Maximum number of clips retrievable per query.

- `max_retrieval_steps`: Maximum number of retrieval steps.

- `multiple_queries`: Whether to use multiple queries.

- `route_switch`: Whether to switch routes if no new clips are retrieved in the last step.

- `planning`: Whether to plan before retrieval.

- `retrieval_threshold`: Threshold for retrieval.

- `logging`: Logging level.

- `save_dir`: Directory for saving generated memory graphs.

- `intermediate_save_dir`: Directory for saving intermediate outputs.

- `input_dir`: Input directory.

- `max_parallel_videos`: Maximum number of videos to process in parallel.

- `log_dir`: Directory for logs.

- `temperature`: Inference temperature.

- `train`: Whether to train.

- `ckpt`: Path to checkpoint.

- `model`: Model to use.

For `configs/memory_config.json`:

- `max_img_embeddings`: Maximum number of faces storable per character in memory.

- `max_audio_embeddings`: Maximum number of voice samples storable per character in memory.

- `img_matching_threshold`: Cosine similarity threshold for considering two faces as belonging to the same individual.

- `audio_matching_threshold`: Cosine similarity threshold for considering two voice samples as belonging to the same individual.

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