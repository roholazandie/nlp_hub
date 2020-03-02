import torch
import os
from config import NLPHubConfig

config = NLPHubConfig.from_json_file("config.json")
os.environ["TORCH_HOME"] = config.torch_home

en2fr = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr',
                       tokenizer='moses',
                       bpe='subword_nmt')

sentences = ["this is a test.", "can you help me?", "This is not great!"]

for sentence in sentences:
    fr = en2fr.translate(sentence, beam=5)
    print(fr)