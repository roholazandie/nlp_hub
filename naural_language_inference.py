import torch
import os
import time
from config import NLPHubConfig

config = NLPHubConfig.from_json_file("config.json")
os.environ["TORCH_HOME"] = config.torch_home


# Download RoBERTa already finetuned for MNLI
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout for evaluation

with torch.no_grad():
    t1 = time.time()
    # Encode a pair of sentences and make a prediction
    tokens = roberta.encode('Kate is so pissed about his reaction.', 'Kate is pleased by his reaction.')
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 0  # contradiction

    # Encode another pair of sentences
    tokens = roberta.encode('He finished his college.', 'He has a degree.')
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 2  # entailment

    tokens = roberta.encode("I love icecream.", "That is so obvious")
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 1 #neutral
    print(prediction) #

    t2 = time.time()
    print(t2-t1)