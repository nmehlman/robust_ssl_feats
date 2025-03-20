from torchvision.datasets import ImageNet
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor

from torchvision.models.resnet import ResNet50_Weights

hugging_face_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

HUGGING_FACE_TRANSFORM = lambda x: hugging_face_processor(x, return_tensors="pt")['pixel_values'].squeeze()



IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(size=224),
        transforms.Normalize(
            mean = (123.675/255, 116.28/255, 103.53/255),
            std = (58.395/255, 57.12/255, 57.375/255),
        )
    ]
)

TORCHVSION_V1_SUPERVISED_TRANSFORM = ResNet50_Weights.IMAGENET1K_V1.transforms()

TORCHVSION_V2_SUPERVISED_TRANSFORM = ResNet50_Weights.IMAGENET1K_V2.transforms()

SIMCLR_SUPERVISED_TRANSFORM = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

IMAGENET_TRANSFORMS = {
    "default": IMAGENET_TRANSFORM, 
    "supervised_torchvision_v1": TORCHVSION_V1_SUPERVISED_TRANSFORM, 
    "supervised_torchvision_v2": TORCHVSION_V2_SUPERVISED_TRANSFORM, 
    "supervised_sim_clr": SIMCLR_SUPERVISED_TRANSFORM,
    "supervised_hugging_face": HUGGING_FACE_TRANSFORM
    }


def get_imagenet_dataloader(
        root: str = '../data/imagenet', 
        split: str = 'val',
        batch_size: int = 1, 
        transform_name: str = "default",
        **loader_kwargs
    ) -> DataLoader:

    transform = IMAGENET_TRANSFORMS[transform_name]

    dset = ImageNet(root=root, split=split, transform=transform)

    data_loader = DataLoader(dset, batch_size=batch_size, shuffle=True, **loader_kwargs)

    return data_loader