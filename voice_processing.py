from utils.chat_api import *
from prompts import *
from utils.general import *
from utils.video_processing import *
import base64
import struct
from laplace import Client

# laplace = Client("sd://lab.agent.audio_embedding_server?idc=maliva&cluster=default", timeout=500)
laplace = Client("tcp://10.124.106.228:9473", timeout=500)

def process_voices(video_graph, base64_audio):
    print(get_audio_info_from_base64(base64_audio))

    input = [
        {
            "type": "audio_base64/wav",
            "content": base64_audio,
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

    return asrs

if __name__ == "__main__":
    # video_path = "/mnt/bn/videonasi18n/longlin.kylin/vlm-agent-benchmarking/data/videos/raw/720p/5 Poor People vs 1 Secret Millionaire.mp4"
    # clip, _, audio = process_video_clip(video_path, 0, 3, 10, video_format="mp4", audio_format="wav")

    # wav_path = 'data/audios/speaker1_a_cn_16k.wav'
    # f = open(wav_path, 'rb')
    # wav_bytes = f.read()
    # audio = base64.b64encode(wav_bytes).decode('utf-8')
    # print(get_audio_info_from_base64(audio))

    wav_path = 'data/audios/tmpi9_79etm.wav'
    f = open(wav_path, 'rb')
    wav_bytes = f.read()
    audio = base64.b64encode(wav_bytes)
    # print(get_audio_info_from_base64(audio))

    # input = [
    #     {
    #         "type": "audio_base64/wav",
    #         "content": audio
    #     },
    #     {
    #         "type": "text",
    #         "content": "describe what you hear from the audio"
    #     }
    # ]
    # messages = generate_messages(input)
    # model = "gemini-1.5-pro-002"
    # response = get_response_with_retry(model, messages)
    # print(response)

    outputs = laplace.matx_inference("audio_embedding", {"wav": [audio]})

    emb = outputs.output_bytes_lists["output"][0]
    format_string = 'f' * (len(emb) // struct.calcsize('f'))
    double_list_recovered = list(struct.unpack(format_string, emb))
    print(double_list_recovered)

    # endpoint_handler = EndpointHandler()
    
    # outputs = endpoint_handler({"wav": [audio]})
    # for emb in outputs["output"]:
    #     format_string = 'f' * (len(emb) // struct.calcsize('f'))
    #     double_list_recovered = struct.unpack(format_string, emb)
    #     # double_list_recovered = struct.unpack('d'*len(emb) // 8, emb)
    #     print(double_list_recovered)