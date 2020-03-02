import json
from pathlib import Path
import os


class NLPHubConfig:
    def __init__(self,
                 torch_home="",
                 ):
        self.torch_home = torch_home

    @classmethod
    def from_dict(cls, json_object):
        config = NLPHubConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]

        if not config.torch_home:
            config.torch_home = os.path.join(str(Path.home()), ".cache/torch")
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))