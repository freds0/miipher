import re
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
from os.path import join, basename,splitext

class BRSpeechCorpus(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = Path(root)
        with open(join(self.root, 'metadata_samples.csv')) as f:
            data = f.readlines()    

        self.wav_files = []
        self.text_data = []
        self.speakers  = []
        for line in data:
            filepath, text, speaker = line.strip().split('|')
            self.wav_files.append(join(self.root, filepath))
            self.text_data.append(text)
            self.speakers.append(speaker)
        

    def __getitem__(self, index):
        wav_path = self.wav_files[index]
        clean_text = self.text_data[index]
        speaker = self.speakers[index]
        filename = basename(wav_path)
        output = {
            "wav_path": str(wav_path),
            "speaker": speaker,
            "clean_text": clean_text,
            "word_segmented_text": clean_text,
            "basename": splitext(filename)[0],
            "lang_code": "pt-br" # https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
            #    "phones": phones
        }

        return output

    def __len__(self):
        return len(self.wav_files)

    @property
    def speaker_dict(self):
        speakers = set()
        for wav_file in self.wav_files:
            basename = splitext(basename(wav_file))[0]
            m = re.search(r"^(\d+?)\_(\d+?)\_(\d+?\_\d+?)$", basename)
            speaker, chapter, utt_id = m.group(1), m.group(2), m.group(3)
            speakers.add(speaker)
        speaker_dict = {x: idx for idx, x in enumerate(speakers)}
        return speaker_dict

    @property
    def lang_code(self):
        return "pt-br"
