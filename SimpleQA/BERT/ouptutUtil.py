import torch
import json


def save_model(model, dicts, modelDir, modelName, dictName):
    torch.save(model.state_dict(), modelDir + "/" + modelName)
    with open(modelDir + "/" + dictName, 'w', encoding='utf-8') as fw:
        fw.write(json.dumps(dicts, ensure_ascii=False))


def load_model(modelDir, modelName, dictName):
    model = torch.load(modelDir + "/" + modelName)
    with open(modelDir + "/" + dictName, 'r', encoding='utf-8') as fr:
        dicts = json.loads(fr.read())
    return model, dicts