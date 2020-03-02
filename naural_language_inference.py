import torch
import os
import time

os.environ["TORCH_HOME"] = "/media/rohola/data/torch_home"

# Download RoBERTa already finetuned for MNLI
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout for evaluation

with torch.no_grad():
    t1 = time.time()
    # Encode a pair of sentences and make a prediction
    tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 0  # contradiction

    # Encode another pair of sentences
    tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.')
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 2  # entailment

    tokens = roberta.encode("I love icecream.", "That is so obvious")
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 1 #neutral
    print(prediction) #

    t2 = time.time()
    print(t2-t1)