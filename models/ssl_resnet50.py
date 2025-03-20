import os

import torch
from torch import nn
from torchvision.models import resnet50

MODEL_WEIGHTS_DIR = os.environ["MODEL_WEIGHTS_DIR"]

def load_resnet50_model(
            weights_name: str,
            weights_dir: str = MODEL_WEIGHTS_DIR,
            num_classes: int = 1000
    ) -> nn.Module:
    """Load pretrained SSL model weights
    
    Args:
        weights_name: (str, optional): name of weights file
        weights_dir (str, optional): path to directory containing saved model weights
        num_classes (int, optional): number of classes

    Returns:
        model (nn.Module): MoCo-trained model
    """

    full_ckpt_path = os.path.join(weights_dir, weights_name)
    state_dict = torch.load(full_ckpt_path)

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    
    model = resnet50(num_classes=num_classes)
    
    model.load_state_dict(state_dict)

    return model

class SSLResNet50Model(nn.Module):

    "Wrapper from SSL ResNet50 Models"

    def __init__(self, weights_name: str, **loader_kwargs):

        super().__init__()
            
        if not weights_name.endswith('.pth'):
            weights_name += '.pth'
        
        self.model = load_resnet50_model(weights_name, **loader_kwargs)
        
        self.model.eval()

        # Create sub-model for feature extraction
        children = list(self.model.children())
        self.feat_model = torch.nn.Sequential(*children[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.model(x)

        return y
    
    def get_features(self, x: torch.Tensor, with_grad: bool = False) -> torch.Tensor:

        if with_grad:
            return self.feat_model(x).squeeze()
        else:
            with torch.no_grad():
                return self.feat_model(x).squeeze()
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        
        with torch.no_grad():
            y = self.model(x)
        
        _, pred = y.topk(1, dim=1)

        return pred.squeeze()