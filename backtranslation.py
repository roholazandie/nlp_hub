import torch
from config import NLPHubConfig
import os

config = NLPHubConfig.from_json_file("config.json")
os.environ["TORCH_HOME"] = config.torch_home

en2de = torch.hub.load('pytorch/fairseq',
                       'transformer.wmt19.en-de.single_model',
                       tokenizer='moses',
                       bpe='fastbpe')

de2en = torch.hub.load('pytorch/fairseq',
                       'transformer.wmt19.de-en.single_model',
                       tokenizer='moses',
                       bpe='fastbpe')

paraphrase = de2en.translate(en2de.translate('PyTorch Hub is an awesome interface!'))

print(paraphrase)
