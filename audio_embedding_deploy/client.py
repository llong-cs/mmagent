import base64
import struct
from laplace import Laplace
laplace = Laplace("lab.agent.audio_embedding_server?idc=maliva&cluster=default", timeout=500)

wav_path = 'speaker1_a_cn_16k.wav'
f = open(wav_path, 'rb')
wav_bytes = f.read()
base64_encoded = base64.b64encode(wav_bytes)
outputs = laplace.matx_inference("audio_embedding", {"wav": [base64_encoded]})

emb = outputs.output_bytes_lists["output"][0]
format_string = 'f' * (len(emb) // struct.calcsize('f'))
double_list_recovered = list(struct.unpack(format_string, emb))
print(double_list_recovered)
