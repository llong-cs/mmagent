import base64
import struct
import json
import os

from laplace import Client
from pydub import AudioSegment
from prompts import prompt_audio_segmentation
from utils.chat_api import generate_messages, get_response_with_retry
from utils.general import validate_and_fix_json, normalize_embedding
from utils.video_processing import process_video_clip
import io

laplace = Client("tcp://10.124.106.228:9473", timeout=500)

processing_config = json.load(open("configs/processing_config.json"))

MAX_RETRIES = processing_config["max_retries"]


def process_voices(video_graph, base64_audio, base64_video, save_path, preprocessing=None):
    def get_audio_segment(base64_audio, start_time, end_time):
        """
        Get audio segment from base64 audio string
        
        Args:
            base64_audio: base64 encoded audio string
            start_time: start time of the audio segment in MM:SS format
            end_time: end time of the audio segment in MM:SS format
        
        Returns:
            tuple: (base64 encoded audio string or None, bool indicating if times are valid)
        """
        # Convert MM:SS to seconds
        try:
            start_min, start_sec = map(int, start_time.split(':'))
            end_min, end_sec = map(int, end_time.split(':'))
        except ValueError:
            return None

        if (start_min < 0 or start_sec < 0 or start_sec >= 60) or (end_min < 0 or end_sec < 0 or end_sec >= 60):
            return None

        start_time_msec = (start_min * 60 + start_sec) * 1000
        end_time_msec = (end_min * 60 + end_sec) * 1000

        if start_time_msec >= end_time_msec:
            return None

        # Decode base64 audio into bytes
        audio_data = base64.b64decode(base64_audio)
        
        # Create BytesIO object to hold audio data
        audio_io = io.BytesIO(audio_data)
        audio = AudioSegment.from_wav(audio_io)

        # Extract segment
        if end_time_msec > len(audio):  # AudioSegment uses milliseconds
            return None
            
        segment = audio[start_time_msec:end_time_msec]
        
        # Export segment to bytes buffer
        segment_buffer = io.BytesIO()
        segment.export(segment_buffer, format='wav')
        segment_buffer.seek(0)
        
        # Convert to base64
        return base64.b64encode(segment_buffer.getvalue())

    def diarize_audio(base64_video):
        input = [
            {
                "type": "video_base64/mp4",
                "content": base64_video.decode("utf-8"),
            },
            {
                "type": "text",
                "content": prompt_audio_segmentation,
            },
        ]
        messages = generate_messages(input)
        model = "gemini-1.5-pro-002"
        asrs = None
        for i in range(MAX_RETRIES):
            response = get_response_with_retry(model, messages)
            asrs = validate_and_fix_json(response[0])
            if asrs is not None:
                break
        if asrs is None:
            raise Exception("Failed to diarize audio")

        for asr in asrs:
            start_min, start_sec = map(int, asr["start_time"].split(':'))
            end_min, end_sec = map(int, asr["end_time"].split(':'))
            asr["duration"] = (end_min * 60 + end_sec) - (start_min * 60 + start_sec)

        return asrs

    def get_normed_audio_embeddings(audios):
        """
        Get normalized audio embeddings for a list of base64 audio strings
        
        Args:
            base64_audios (list): List of base64 encoded audio strings
            
        Returns:
            list: List of normalized audio embeddings
        """
        audio_segments = [audio["audio_segment"] for audio in audios]
        outputs = laplace.matx_inference("audio_embedding", {"wav": audio_segments})
        embeddings = outputs.output_bytes_lists["output"]
        normed_embeddings = [normalize_embedding(embedding) for embedding in embeddings]
        for audio, embedding in zip(audios, normed_embeddings):
            audio["embedding"] = embedding
        return audios

    # TODO: segment all at once, and filtering can go first
    def create_audio_segments(base64_audio, asrs):
        for asr in asrs:
            if "audio_segment" not in asr:
                start_time = asr["start_time"]
                end_time = asr["end_time"]
                audio_segment = get_audio_segment(base64_audio, start_time, end_time)
                asr["audio_segment"] = audio_segment

        return asrs
    
    # TODO: ordering while mapping
    def establish_mapping(asrs, key="speaker"):
        """
        Establish mapping between audio segments and characters based on ASR results
        
        Args:
            asrs (list): List of ASR results
        
        Returns:
            dict: Mapping of audio segments to characters, with segments sorted by duration
        """
        mapping = {}
        if key not in asrs[0]:
            raise ValueError(f"Key {key} not found in ASR results")

        for asr in asrs:

            id = asr[key]
            if id not in mapping:
                mapping[id] = []
            mapping[id].append(asr)

        # Sort segments for each speaker by duration
        for id in mapping:
            # Filter out entries with None audio_segment first, then sort by duration
            mapping[id] = sorted([x for x in mapping[id] if x["audio_segment"] is not None], 
                                   key=lambda x: x["duration"], 
                                   reverse=True)

        # Filter out speakers with no segments
        mapping = {k: v for k, v in mapping.items() if v}

        return mapping

    def filter_duration_based(audios):
        min_duration = processing_config["min_duration_for_audio"]
        filtered_audios = [
            audio
            for audio in audios
            if audio["duration"] >= min_duration
        ]
        return filtered_audios
    
    # def update_videograph(video_graph, tempid2audios, filter=None):
    #     audios_list = []
    #     for tempid, audios in tempid2audios.items():
    #         if filter:
    #             filtered_audios = filter(audios)
    #         else:
    #             filtered_audios = audios
    #         voice_embs = [audio["embedding"] for audio in filtered_audios]
    #         if len(voice_embs) == 0:
    #             continue
    #         else:
    #             matched_nodes = video_graph.search_voice_nodes(voice_embs)
    #             if len(matched_nodes) > 0:
    #                 matched_node = matched_nodes[0][0]
    #                 video_graph.add_embedding(matched_node, voice_embs)
    #                 for audio in audios:
    #                     audio["matched_node"] = matched_node
    #             else:
    #                 matched_node = video_graph.add_voice_node(voice_embs)
    #                 for audio in audios:
    #                     audio["matched_node"] = matched_node

    #         audios_list.extend(audios)

    #     return audios_list

    # asrs = diarize_audio(base64_video)
    # print(asrs)
    # tempid2audios = establish_mapping(asrs, key="speaker")

    # for _, audios in tempid2audios.items():
    #     audio_segments = [audio["audio_segment"] for audio in audios]
    #     embeddings = get_normed_audio_embeddings(audio_segments)
    #     for audio, embedding in zip(audios, embeddings):
    #         audio["embedding"] = embedding
    

    # audios_list = update_videograph(video_graph, tempid2audios, filter=filter_duration_based)
    # id2audios = establish_mapping(audios_list, key="matched_node")

    # return id2audios
    
    def update_videograph(video_graph, audios, filter=None):
        audios_list = []
        if filter:
            filtered_audios = filter(audios)
        else:
            filtered_audios = audios
        for audio in filtered_audios:
            audio_info = {
                "embeddings": [audio["embedding"]],
                "contents": [audio["asr"]]
            }
            matched_nodes = video_graph.search_voice_nodes(audio_info)
            if len(matched_nodes) > 0:
                matched_node = matched_nodes[0][0]
                video_graph.update_node(matched_node, audio_info)
                audio["matched_node"] = matched_node
            else:
                matched_node = video_graph.add_voice_node(audio_info)
                audio["matched_node"] = matched_node

            audios_list.append(audio)

        return audios_list

    # Check if intermediate results exist
    try:
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                audios = json.load(f)
            for audio in audios:
                audio["audio_segment"] = audio["audio_segment"].encode("utf-8")
        else:
            asrs = diarize_audio(base64_video)
            audios = create_audio_segments(base64_audio, asrs)
            audios = [audio for audio in audios if audio["audio_segment"] is not None]

            if len(audios) != 0:
                audios = get_normed_audio_embeddings(audios)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "w") as f:
                for audio in audios:
                    audio["audio_segment"] = audio["audio_segment"].decode("utf-8")
                json.dump(audios, f)
                for audio in audios:
                    audio["audio_segment"] = audio["audio_segment"].encode("utf-8")
            
            print(f"Write voice detection results to {save_path}")
    except Exception as e:
        if preprocessing:
            # Save error to log file
            log_dir = processing_config["log_dir"]
            os.makedirs(log_dir, exist_ok=True)
            error_log_path = os.path.join(log_dir, "error_voice_preprocessing.log")
            with open(error_log_path, "a") as f:
                f.write(f"Error processing {save_path}: {str(e)}\n")
        raise RuntimeError(f"Failed to diarize audio at {save_path}: {e}")
    
    if preprocessing:
        return {}
    
    if len(audios) == 0:
        return {}

    audios_list = update_videograph(video_graph, audios, filter=filter_duration_based)
    id2audios = establish_mapping(audios_list, key="matched_node")

    return id2audios

if __name__ == "__main__":
    video_path = "/mnt/bn/videonasi18n/longlin.kylin/vlm-agent-benchmarking/data/videos/raw/720p/5 Poor People vs 1 Secret Millionaire.mp4"
    clip, _, audio = process_video_clip(video_path, 0, 3, 10, audio_format="wav")

    # print(audio)

    outputs = laplace.matx_inference("audio_embedding", {"wav": [audio, audio]})

    emb = outputs.output_bytes_lists["output"][0]
    format_string = 'f' * (len(emb) // struct.calcsize('f'))
    double_list_recovered = list(struct.unpack(format_string, emb))
    print(len(double_list_recovered))
