from typing import List, Dict, Union
import struct
import torch
import torchaudio
import base64
from io import BytesIO
from speakerlab.process.processor import FBank
from speakerlab.utils.builder import dynamic_import

class EndpointHandler:
    def __init__(self) -> None:
        self.device = torch.device('cuda')
        pretrained_state = torch.load("pretrained_eres2netv2.ckpt", map_location='cpu')
        model = {
            'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
            'args': {
                'feat_dim': 80,
                'embedding_size': 192,
            },
        }
        self.embedding_model = dynamic_import(model['obj'])(**model['args'])
        self.embedding_model.load_state_dict(pretrained_state)
        self.embedding_model.to(self.device)
        self.embedding_model.eval()
        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        print("Finish load model.")
    

    def get_embedding(self, wav):

        def load_wav(wav_file, obj_fs=16000):
            wav, fs = torchaudio.load(wav_file)
            if fs != obj_fs:
                wav, fs = torchaudio.sox_effects.apply_effects_tensor(
                    wav, fs, effects=[['rate', str(obj_fs)]]
                )
            if wav.shape[0] > 1:
                wav = wav[0, :].unsqueeze(0)
            return wav
        
        def compute_embedding(wav_file, save=True):
            wav = load_wav(wav_file)
            feat = self.feature_extractor(wav).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.embedding_model(feat).detach().squeeze(0).cpu().numpy()
            return embedding

        return compute_embedding(wav)

    @torch.no_grad()
    def generate(self, wav):
        wav = base64.b64decode(wav)
        wav_file = BytesIO(wav)
        emb = self.get_embedding(wav_file)
        return emb

    @torch.no_grad()
    def __call__(
        self, request: Dict[str, Union[List[bytes], List[int], List[float]]]
    ) -> Dict[str, Union[List[bytes], List[int], List[float]]]:
        res = []
        for wav in request["wav"]:
            # completion = self.generate(wav.decode("utf-8"))
            completion = self.generate(wav)
            print(completion)
            # bytes_data = b''.join(struct.pack('d', num) for num in completion)
            bytes_data = struct.pack('f' * len(completion), *completion)
            res.append(bytes_data)
            
        return dict(output=res)

if __name__ == "__main__":
    endpoint_handler = EndpointHandler()
    
    wav_path = 'speaker1_a_cn_16k.wav'
    with open(wav_path, 'rb') as f:
        wav_bytes = f.read()
        base64_encoded = base64.b64encode(wav_bytes)
        outputs = endpoint_handler({"wav": [base64_encoded]})
        for emb in outputs["output"]:
            format_string = 'f' * (len(emb) // struct.calcsize('f'))
            double_list_recovered = struct.unpack(format_string, emb)
            # double_list_recovered = struct.unpack('d'*len(emb) // 8, emb)
            print(double_list_recovered)
