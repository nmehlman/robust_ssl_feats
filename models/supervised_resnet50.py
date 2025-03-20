import sys
import os
import torch
from torch import nn
from torchvision.models import resnet50
from contextlib import nullcontext
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch.nn.functional as F

SIM_CLR_PYTORCH_DIR = os.environ["SIM_CLR_PYTORCH_DIR"]
MODEL_WEIGHTS_DIR = os.environ["MODEL_WEIGHTS_DIR"]

def load_sim_clr_supervised_resnet(
        weights_name: str = 'r50_1x_sk0', 
        sim_clr_pytorch_dir: str = SIM_CLR_PYTORCH_DIR,
        weights_dir: str = MODEL_WEIGHTS_DIR
    ) -> nn.Module:

    """Loads supervised ResNet models from SimCLRv2 repo
    
    Args:
        weights_name (str, optional): name of pretrained model (format: r<size>_<width>x_sk<0/1>)
        sim_clr_pytorch_dir (str, optional): path to SimCLRv2-Pytorch repo directory for converting TF models to PyTorch.
        weights_dir (str, optional): path to directory containing saved model weights
        
    Returns:
        model (nn.Model): restored PyTorch model
    """

    # Required imports from SimCLRv2-Pytorch
    sys.path.append(sim_clr_pytorch_dir)
    from download import available_simclr_models, simclr_categories
    from resnet import get_resnet, name_to_params 

    full_ckpt_path = os.path.join(weights_dir, f"{weights_name}.pth")

    # Create ResNet model
    model, _ = get_resnet(*name_to_params(full_ckpt_path.split('/')[-1]))

    # Load state dict to model
    state_dict = torch.load(full_ckpt_path)
    model.load_state_dict(state_dict['resnet'])

    return model

def load_torchvision_supervised_resnet(weights_name: str):

    return resnet50(weights=weights_name)


class SupervisedResNet50Model(nn.Module):

    "Wrapper for supervised ResNet50 Models"

    def __init__(self, model_type: str, weights_name: str = None):

        super().__init__()

        self.model_type = model_type
        if model_type == 'torchvision':
            self.model = load_torchvision_supervised_resnet(weights_name=weights_name)
            
            # Create sub-model for feature extraction
            children = list(self.model.children())
            self.feat_model = torch.nn.Sequential(*children[:-1])

        elif model_type == 'sim_clr':
            self.model = load_sim_clr_supervised_resnet(weights_name=weights_name)

        elif model_type == 'hugging_face':
            self.model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

        else:
            raise ValueError('invalid model type')

        self.model.eval()


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.model_type == 'torchvision':
            y = self.model(x)

        elif self.model_type == 'sim_clr':
            y = self.model(x, apply_fc=True)

        elif self.model_type == 'hugging_face':
            y = self.model(x).logits

        return y
    
    def get_features(self, x: torch.Tensor, with_grad: bool = False) -> torch.Tensor:
        
        context = nullcontext() if with_grad else torch.no_grad()

        with context:
            if self.model_type == 'torchvision':
                z = self.feat_model(x).squeeze()

            elif self.model_type == 'sim_clr':
                z = self.model(x, apply_fc=False).squeeze()

            elif self.model_type == 'hugging_face':
                z = self.model(x, output_hidden_states=True).hidden_states[-1]
                z = self.model.resnet.pooler(z).squeeze()

        return z
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        
        with torch.no_grad():
            if self.model_type == 'torchvision':
                y = self.model(x)

            elif self.model_type == 'sim_clr':
                y = self.model(x, apply_fc=True)

            elif self.model_type == 'hugging_face':
                y = self.model(x).logits

        
        _, pred = y.topk(1, dim=1)

        return pred

    
if __name__ == "__main__":

    from test_code.utils import load_imagenet_sample
    from data import HUGGING_FACE_TRANSFORM
    hugging_face_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    x,y = load_imagenet_sample(batch_size=16, transform=HUGGING_FACE_TRANSFORM)

    model = SupervisedResNet50Model(model_type='hugging_face')
    
    x.requires_grad = True
    y_pred = model.predict(x)
    print(y_pred.shape)

