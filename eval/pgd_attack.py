from typing import Any, Optional, Tuple, Union
from art.estimators.classification.pytorch import PyTorchClassifier
from numpy import ndarray
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch

def generate_random_targets(nb_classes: int, y_pred: np.ndarray) -> np.ndarray:
    """Helper function to generate random target classes based on model's clean predictions

    Args:
        nb_classes (int): number of classes
        y_pred (np.ndarray): original predictions to be excluded as targets

    Returns:
        y_tgt (np.ndarray): target classes
    """
    
    batch_size = y_pred.shape[0]

    # Initialize the result array
    result = np.empty(batch_size, dtype=int)

    # Iterate over each position
    for i in range(batch_size):
        # Generate a range of values excluding the current value in excluded_values
        possible_values = np.setdiff1d(np.arange(nb_classes), np.array([y_pred[i]]))
        
        # Randomly select one of the possible values
        result[i] = np.random.choice(possible_values)

    return result


class PGD:

    """Class for Projected Gradient Descent (PGD) attack based on ART implementation
    https://github.com/Trusted-AI/adversarial-robustness-toolbox
    """

    def __init__(self, 
                model: nn.Module,
                input_shape: Tuple[int] = (224,224,3),
                norm: Union[int, float, str] = np.inf,
                eps: Union[int, float, np.ndarray] = 0.3,
                eps_step: Union[int, float, np.ndarray] = 0.1,
                decay: Optional[float] = None,
                max_iter: int = 100,
                targeted: bool = False,
                num_random_init: int = 0,
                batch_size: int = 32,
                verbose: bool = False,
                nb_classes: int = 1000,
                **kwargs
        ) -> None:

        """Initilaized PGD attack instance
        Args:
            model (ModelWrapper): model to be attacked
            input_shape (Tuple[int]): shape of one input sample
            norm (Union[str, np.inf, int], optional): the norm of the adversarial perturbation. Possible values: "inf", np.inf, 1, or 2.
            eps (float, optional): maximum perturbation that the attacker can introduce.
            eps_step (float, optional): attack step size (input variation) at each iteration.
            random_eps (bool, optional): when True, epsilon is drawn randomly from truncated normal distribution. Suggested for FGSM based training to generalize across different epsilons. eps_step is modified to preserve the ratio of eps / eps_step. The effectiveness of this method with PGD is untested (https://arxiv.org/pdf/1611.01236.pdf).
            max_iter (int, optional): the maximum number of iterations.
            targeted (bool, optional): indicates whether the attack is targeted (True) or untargeted (False).
            num_random_init (int, optional): number of random initialisations within the epsilon ball. For num_random_init=0, starting at the original input.
            batch_size (int, optional): size of the batch on which adversarial samples are generated.
            summary_writer (Union[bool, str, SummaryWriter], optional): activate summary writer for TensorBoard. Default is `False` and deactivated summary writer. If `True`, saves runs/CURRENT_DATETIME_HOSTNAME in the current directory. If of type `str`, saves in the path. If of type `SummaryWriter`, applies the provided custom summary writer. Use hierarchical folder structure to compare between runs easily, e.g., ‘runs/exp1’, ‘runs/exp2’, etc., for each new experiment to compare across them.
            verbose (bool, optional): show progress bars.
            nb_classes (int, optional): number of classes in dataset. Defaults to 1000 for ImageNet
        """
        
        loss = nn.CrossEntropyLoss()

        self.nb_classes = nb_classes
        self.targeted = targeted

        self.art_wrapped_model = PyTorchClassifier( # Needed to interface with ART PGD attack
                                                model = model, 
                                                loss = loss, 
                                                input_shape = input_shape, 
                                                nb_classes = nb_classes, 
                                                **kwargs
                                                )

        self.attack = ProjectedGradientDescentPyTorch( # ART PGD implementation
                                                estimator = self.art_wrapped_model,
                                                norm = norm,
                                                eps = eps,
                                                eps_step = eps_step,
                                                decay = decay,
                                                max_iter = max_iter,
                                                targeted = targeted,
                                                num_random_init = num_random_init,
                                                batch_size = batch_size,
                                                verbose = verbose,
                                            )
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        """Generated attack for sample x
        
        Args:
            x (torch.Tensor): image tensor
        
        Returns:
            x_adv (torch.Tensor): adversarial image 
        """

        x = x.cpu().numpy()

        if self.targeted: # Randomly select class as target excluding model's clean predictions
            y_pred = self.art_wrapped_model.predict(x).argmax(-1)
            y_tgt = generate_random_targets(self.nb_classes, y_pred)
        else:
            y_tgt = None

        x_adv = self.attack.generate(x, y_tgt)

        return torch.from_numpy(x_adv)

if __name__ == "__main__":

    from test_code.utils import load_imagenet_sample
    from models.sim_clr import SIM_CLR_TRANSFORM, SimCLRv2
    from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch

    x, y = load_imagenet_sample(transform=SIM_CLR_TRANSFORM)
    x = x.cuda()

    model = SimCLRv2()

    attack = PGD(model=model, input_shape=x.shape[1:], targeted=True)
    
    x_adv = attack(x)
    
    y_pred_clean = model.predict(x.cuda()).item()
    y_pred_adv = model.predict(x_adv.cuda()).item()
    y_clean = y.item()
    
    print(f"True label: {y_clean}\nClean predicted label: {y_pred_clean}\nAdv. predicted label: {y_pred_adv}")

   
    
