import argparse
import string
from transformers import AutoTokenizer
#from text_normalize import normalize_text, remove_accents
from miipher.ml_pl_bert.text_normalize import normalize_text, remove_accents
#from miipher.ml_pl_bert.text_utils import TextCleaner
import phonemizer

import string
from nltk.tokenize import TweetTokenizer
nltk_tokenizer = TweetTokenizer()

from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()

def word_tokenize(text): return nltk_tokenizer.tokenize(text)
def detokenize(tokens): return detokenizer.detokenize(tokens)

def generate_trigrams(tokens):
    trigrams = []
    for i in range(len(tokens) - 2):
        trigram = tokens[i:i + 3]
        trigrams.append(trigram)
    return trigrams

class PhonemizeProcessor:

    def __init__(self, language='en-us', preserve_punctuation=True, with_stress=True):
        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language=language,
            preserve_punctuation=preserve_punctuation,
            with_stress=with_stress
        )
        #self.text_cleaner = TextCleaner() 
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")

    def _phonemize(self, text):
        words = word_tokenize(text)
        trigrams = generate_trigrams(words)
        if len(trigrams) == 0:
            print("Empty trigram...")      
            return {'input_ids' : None, 'phonemes': None}
        pairs = []
        
        k = trigrams[0]
        trigram = detokenize(k)
        phonemes = word_tokenize(self.global_phonemizer.phonemize([trigram], strip=True)[0])

        word = k[0]

        if len(phonemes) == 3:
            pairs.append((self.tokenizer.encode(word)[1:-1], phonemes[0]))
        else:
            pairs.append((self.tokenizer.encode(word)[1:-1], self.global_phonemizer.phonemize([k[0]], strip=True)[0]))

        for k in trigrams:
            trigram = detokenize(k)
            word = k[1]
            if k[1] in string.punctuation:
                pairs.append((self.tokenizer.encode(word)[1:-1], word))
                continue
            phonemes = word_tokenize(self.global_phonemizer.phonemize([trigram], strip=True)[0])

            if len(phonemes) == 3:
                pairs.append((self.tokenizer.encode(word)[1:-1], phonemes[1]))
            else:
                pairs.append((self.tokenizer.encode(word)[1:-1], self.global_phonemizer.phonemize([k[1]], strip=True)[0]))

        k = trigrams[-1]
        trigram = detokenize(k)
        phonemes = word_tokenize(self.global_phonemizer.phonemize([trigram], strip=True)[0])

        word = k[-1]
        if len(phonemes) == 3:
            pairs.append((self.tokenizer.encode(word)[1:-1], phonemes[-1]))
        else:
            pairs.append((self.tokenizer.encode(word)[1:-1], self.global_phonemizer.phonemize([k[-1]], strip=True)[0]))


        input_ids = []
        phonemes = []

        for p in pairs:
            input_ids.append(p[0])
            phonemes.append(p[1])
        assert len(input_ids) == len(phonemes)   
        return {'input_ids' : input_ids, 'phonemes': phonemes}

    
    def __call__(self, text):
        return self._phonemize(text)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pl_bert_dir", default="src/miipher/ml_pl_bert/checkpoints")

    args = parser.parse_args()

    input_text = "It is not in the stars to hold our destiny but in ourselves; we are underlings."
    preprocessor = PhonemizeProcessor(language='en-us')
    values = preprocessor(input_text)

    print(input_text)
    print(values['input_ids'])
    print(values['phonemes'])
