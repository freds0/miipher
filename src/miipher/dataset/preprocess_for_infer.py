import torch 
import hydra
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from miipher.text_encoder.phonemize_processor import PhonemizeProcessor

class PreprocessForInfer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.speech_ssl_processor = hydra.utils.instantiate(
            cfg.data.speech_ssl_processor.processor
        )
        self.speech_ssl_sr = cfg.data.speech_ssl_processor.sr
        self.cfg = cfg
        self.text2phone_dict = dict()

    @torch.inference_mode()
    def get_phonemes_input_ids(self, word_segmented_text, lang_code):
        if lang_code not in self.text2phone_dict.keys():
            self.text2phone_dict[lang_code] = PhonemizeProcessor(language=lang_code)
            
        input_ids, input_phonemes = self.text2phone_dict[lang_code](word_segmented_text)
        return input_ids, input_phonemes        

    
    def process(self, basename, degraded_audio,word_segmented_text=None,lang_code=None, phoneme_text=None):
        degraded_audio, sr = degraded_audio
        output = dict()

        if word_segmented_text != None and  lang_code != None:
            input_ids, input_phonems = self.get_phonemes_input_ids(
                word_segmented_text, lang_code
            )
            input_ids_pt = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
            output['phoneme_input_ids'] = input_ids_pt
        #elif phoneme_text == None:
        else:
            raise ValueError

        degraded_16k = torchaudio.functional.resample(
            degraded_audio, sr, new_freq=16000
        ).squeeze()
        degraded_wav_16ks = [degraded_16k]

        output["degraded_ssl_input"] = self.speech_ssl_processor(
            [x.cpu().numpy() for x in degraded_wav_16ks],
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
        )
        output["degraded_wav_16k"] = pad_sequence(degraded_wav_16ks, batch_first=True)
        output["degraded_wav_16k_lengths"] = torch.tensor(
            [degraded_wav_16k.size(0) for degraded_wav_16k in degraded_wav_16ks]
        )
        return output

