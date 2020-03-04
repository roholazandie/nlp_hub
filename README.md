## NLP Hub
This repo contains the code to perform the basic to most complext NLP tasks in a few lines of code
This is mostly based on [fairseq](https://github.com/pytorch/fairseq) and [transformers](https://github.com/huggingface/transformers).

## Install
to install:
```
pip install -r requirements.txt
```
Add the torch hub home in the config.json. If you leave it empty then we choose `/home/usrname/.cache/torch` for you.

WARNING: This is a hub, so it takes at least a few gigabytes of the disk space. Choose a free partition of your disk to hub home.


## Todos
Add Intent classification using bert(from transformer)