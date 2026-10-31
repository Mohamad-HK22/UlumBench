import json
from importlib.resources import files

def load_taxonomy():
    return json.loads(files("ulumbench.scoring").joinpath("units.json").read_text(encoding='utf-8'))
