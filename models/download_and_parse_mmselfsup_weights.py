import os
import sys
import urllib.request
import torch
from torchvision.models import resnet50

if __name__ == "__main__":
    
    url = sys.argv[1]
    
    prefix = url.split('/')[5]
    name = url.split('/')[-1].replace('.pth', '')

    urllib.request.urlretrieve(url, "temp.pth")

    sd = torch.load('temp.pth')

    state_dict = {}
    for key, val in sd['state_dict'].items():
        if 'backbone' in key:
            state_dict[key.replace('backbone.', '')] = val
        elif 'head' in key:
            state_dict[key.replace('head.', '')] = val

    model = resnet50()
    model.load_state_dict(state_dict) # Testing

    torch.save(state_dict, os.path.join("/work_new/nmehlman/robust_features/models/weights_files", f"{prefix}_{name}_torchvision.pth"))

    os.remove('temp.pth')

