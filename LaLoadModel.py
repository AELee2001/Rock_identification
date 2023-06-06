import torch
import timm
from segment_anything import sam_model_registry


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():

    model_paths = [ './models/final_mobilenetv3.ckpt', './models/final_inception_v3.ckpt', './models/final_DEIT.ckpt', './models/base_DEIT.ckpt']
    model_names = ['tf_mobilenetv3_small_075', 'inception_v3', 'deit_small_patch16_224', 'deit3_base_patch16_224']
    num_classes = [5,5,5,6]

    models = []

    for model_name, ckpt_path, num_class in zip(model_names, model_paths, num_classes):
        print('load {} ing...'.format(model_name))
        model = timm.create_model(model_name, pretrained=False, num_classes=num_class)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    return models

mymodels = load_model()

def load_SAM():
    print('load SAM ing...')

    sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return sam

sam = load_SAM()