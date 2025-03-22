import base64
import struct
import tempfile

import numpy as np
from laplace import Client
from moviepy import AudioFileClip
from pydub import AudioSegment
from prompts import prompt_audio_diarization
from utils.chat_api import generate_messages, get_response_with_retry
from utils.general import validate_and_fix_json
from utils.video_processing import process_video_clip
import io
# laplace = Client("sd://lab.agent.audio_embedding_server?idc=maliva&cluster=default", timeout=500)
laplace = Client("tcp://10.124.106.228:9473", timeout=500)

def process_audio_base64(base64_audio, target_fps=16000, audio_format='wav'):
    """
    Process audio base64 string to binary with target frame rate and format
    Args:
        base64_audio: base64 encoded audio string
        target_fps: target frame rate, default 16000
        audio_format: target audio format (e.g. 'wav', 'mp3', etc), default 'wav'
    Returns:
        Binary audio data with target frame rate
    """
    try:
        # Create temp files for processing
        with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=True) as temp_in, \
             tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=True) as temp_out:

            # Write base64 to temp input file
            audio_data = base64.b64decode(base64_audio)
            temp_in.write(audio_data)
            temp_in.flush()

            # Load and resample audio using moviepy
            audio = AudioFileClip(temp_in.name)
            resampled_audio = audio.set_fps(target_fps)

            # Set audio codec based on format
            if audio_format == 'mp3':
                audio_codec = 'libmp3lame'
            elif audio_format == 'wav':
                audio_codec = 'pcm_s16le'
            else:
                audio_codec = 'libmp3lame'  # Default to mp3

            # Write resampled audio to temp output file
            resampled_audio.write_audiofile(temp_out.name, fps=target_fps, codec=audio_codec, logger=None)

            # Read binary data
            temp_out.seek(0)
            binary_audio = temp_out.read()

            audio.close()
            resampled_audio.close()

            return binary_audio

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise

def normalize_embedding(embedding):
    """Normalize embedding to unit length."""
    format_string = 'f' * (len(embedding) // struct.calcsize('f'))
    emb = np.array(struct.unpack(format_string, embedding))
    norm = np.linalg.norm(emb)
    return (emb / norm).tolist() if norm > 0 else emb.tolist()

def get_normed_audio_embeddings(base64_audios):
    """
    Get normalized audio embeddings for a list of base64 audio strings
    
    Args:
        base64_audios (list): List of base64 encoded audio strings
        
    Returns:
        list: List of normalized audio embeddings
    """
    print(type(base64_audios))
    print(type(base64_audios[0]))
    print(base64_audios)
    outputs = laplace.matx_inference("audio_embedding", {"wav": base64_audios})
    print(outputs)
    embeddings = outputs.output_bytes_lists["output"]
    normed_embeddings = [normalize_embedding(embedding) for embedding in embeddings]
    return normed_embeddings


def process_voices(video_graph, base64_audio):
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

        if start_min < 0 or start_sec < 0 or start_sec >= 60:
            return None
        if end_min < 0 or end_sec < 0 or end_sec >= 60:
            return None

        start_time_sec = start_min * 60 + start_sec
        end_time_sec = end_min * 60 + end_sec

        if start_time_sec >= end_time_sec:
            return None

        # Decode base64 audio into bytes
        audio_data = base64.b64decode(base64_audio)
        
        # Create BytesIO object to hold audio data
        audio_io = io.BytesIO(audio_data)
        audio = AudioSegment.from_wav(audio_io)

        # Extract segment
        if end_time_sec * 1000 > len(audio):  # AudioSegment uses milliseconds
            return None
            
        segment = audio[start_time_sec * 1000:end_time_sec * 1000]
        
        # Export segment to bytes buffer
        segment_buffer = io.BytesIO()
        segment.export(segment_buffer, format='wav')
        segment_buffer.seek(0)
        
        # Convert to base64
        return base64.b64encode(segment_buffer.getvalue())

    def diarize_audio(base64_audio):
        asrs = None
        count = 0
        while not asrs:
            count += 1
            print(f"Diarizing audio {count} times")
            if count > 3:
                raise Exception("Failed to diarize audio")
            input = [
                {
                    "type": "audio_base64/wav",
                    "content": base64_audio.decode("utf-8"),
                },
                {
                    "type": "text",
                    "content": prompt_audio_diarization,
                },
            ]
            messages = generate_messages(input)
            model = "gemini-1.5-pro-002"
            response = get_response_with_retry(model, messages)

            asrs = validate_and_fix_json(response[0])

        for asr in asrs:
            start_min, start_sec = map(int, asr["start_time"].split(':'))
            end_min, end_sec = map(int, asr["end_time"].split(':'))
            asr["duration"] = (end_min * 60 + end_sec) - (start_min * 60 + start_sec)

        return asrs

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
            if "audio_segment" not in asr:
                start_time = asr["start_time"]
                end_time = asr["end_time"]
                audio_segment = get_audio_segment(base64_audio, start_time, end_time)
                asr["audio_segment"] = audio_segment

            if asr[key] not in mapping:
                mapping[asr[key]] = []
            mapping[asr[key]].append(asr)

        # Sort segments for each speaker by duration
        for speaker in mapping:
            # Filter out entries with None audio_segment first, then sort by duration
            mapping[speaker] = sorted([x for x in mapping[speaker] if x["audio_segment"] is not None], 
                                   key=lambda x: x["duration"], 
                                   reverse=True)

        return mapping

    def filter_duration_based(audios):
        min_duration = 2
        max_voices = 3
        filtered_audios = [
            audio
            for audio in audios
            if audio["duration"] >= min_duration
        ]
        return filtered_audios[:max_voices]

    def update_videograph(video_graph, tempid2audios, filter=None):
        audios_list = []
        for tempid, audios in tempid2audios.items():
            if filter:
                filtered_audios = filter(audios)
            else:
                filtered_audios = audios
            filtered_embeddings = [audio["embedding"] for audio in filtered_audios]
            matched_nodes = video_graph.search_voice_nodes(filtered_embeddings)
            if len(matched_nodes) > 0:
                matched_node = matched_nodes[0][0]
                video_graph.add_embedding(matched_node, filtered_embeddings)
                for audio in audios:
                    audio["matched_node"] = matched_node
            else:
                matched_node = video_graph.add_voice_node(filtered_embeddings)
                for audio in audios:
                    audio["matched_node"] = matched_node
            audios_list.extend(filtered_audios)

        return audios_list

    asrs = diarize_audio(base64_audio)
    print(asrs)
    tempid2audios = establish_mapping(asrs, key="speaker")
    for _, audios in tempid2audios.items():
        audio_segments = [audio["audio_segment"] for audio in audios]
        embeddings = get_normed_audio_embeddings(audio_segments)
        for audio, embedding in zip(audios, embeddings):
            audio["embedding"] = embedding

    audios_list = update_videograph(video_graph, tempid2audios, filter_duration_based)
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
