from typing import Dict, Tuple
from models.ssl_resnet50 import SSLResNet50Model
from models.supervised_resnet50 import SupervisedResNet50Model
from data import get_imagenet_dataloader
from pgd_attack import PGD
import torch
import json
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader

import tqdm
import os
import yaml

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

ATTACK_INFO = {"pgd": PGD}

class RunningMean:

    """Helper class for computing running mean batch-by-batch"""

    def __init__(self, shape: tuple):

        self.value = np.zeros(shape=shape)
        self.counter = 0

    def update(self, new_value: np.ndarray, num_samples: int = 1):

        self.value = self.counter/(self.counter + num_samples) * self.value \
                    + num_samples/(self.counter + num_samples) * new_value
        
        self.counter += num_samples

    def get_value(self):
        return self.value

class RobustnessTracker:

    """Helper class to keep track of information required for robustness evaluation"""

    def __init__(self, model: nn.Module, nb_classes: int, nb_features: int):

        """Creates new instance of RobustnessTracker
        
        Args:
            model (ModelWrapper): model under testing
            nb_classes (int): number of classes
            nb_features (int): number of features
        """

        self.model = model
        self.nb_classes = nb_classes
        self.nb_features = nb_features

        # These matrices will hold the running estimate of E[Z]
        self.mu_z_clean = RunningMean((nb_features,))
        self.mu_z_adv = RunningMean((nb_features,))

        # These matricies will hold the running estimate of E[Z^2]
        self.mu_z2_clean = RunningMean((nb_features,))
        self.mu_z2_adv = RunningMean((nb_features,))

        # These matrices will hold the running estimate of E[IZ]
        self.mu_iz_clean = RunningMean((nb_classes, nb_features))
        self.mu_iz_adv = RunningMean((nb_classes, nb_features))

        # Will hold the number of sample in each class
        self.class_counts = np.zeros((nb_classes))

        # Will hold model's predictions
        self.clean_predictions = []
        self.adv_predictions = []
        self.true_labels = []

    def update(self, x: torch.tensor, x_adv: torch.tensor, y: torch.tensor):

        """Updates tracking for batch
        
        Args:
            x (torch.tensor): batch clean samples
            x_adv (torch.tensor): batch adversarial samples
            y (torch.tensor): true batch labels
        """

        batch_samples = x.shape[0]

        # Get model predictions
        y_pred_clean = self.model.predict(x.cuda())
        y_pred_adv = self.model.predict(x_adv.cuda())

        # Store predictions and true labels
        self.clean_predictions += y_pred_clean.squeeze().tolist()
        self.adv_predictions += y_pred_adv.squeeze().tolist()
        self.true_labels += y.squeeze().tolist()
        
        # Get features
        z_clean = self.model.get_features(x.cuda()).cpu().numpy()
        z_adv = self.model.get_features(x_adv.cuda()).cpu().numpy()
        
        y = y.cpu().numpy()
        x = x.cpu()
        x_adv = x_adv.cpu()

        # Update feature matrices
        self.mu_z_clean.update(z_clean.mean(0), num_samples=batch_samples)
        self.mu_z_adv.update(z_adv.mean(0), num_samples=batch_samples)

        self.mu_z2_clean.update( (z_clean**2).mean(0), num_samples=batch_samples)
        self.mu_z2_adv.update( (z_adv**2).mean(0), num_samples=batch_samples)

        mu_iz_clean_batch = np.zeros((self.nb_classes, self.nb_features))
        np.add.at(mu_iz_clean_batch, y, z_clean)
        self.mu_iz_clean.update(mu_iz_clean_batch/batch_samples, num_samples=batch_samples)

        mu_iz_adv_batch = np.zeros((self.nb_classes, self.nb_features))
        np.add.at(mu_iz_adv_batch, y, z_adv)
        self.mu_iz_adv.update(mu_iz_adv_batch/batch_samples, num_samples=batch_samples)

        np.add.at(self.class_counts, y, 1)

    def compute_classifier_robustness(self) -> Dict[str, float]:
        """Computes classifier-level robustness metrics
        
        Returns:
            metrics (dict): dict of robustness metrics 
        """

        total_samples = len(self.true_labels)

        assert len(self.adv_predictions) == len(self.clean_predictions) == total_samples

        true_labels = np.array(self.true_labels)
        clean_predictions = np.array(self.clean_predictions)
        adv_predictions = np.array(self.adv_predictions)

        is_correct_clean = (clean_predictions == true_labels)
        is_correct_adv = (adv_predictions == true_labels)
        is_faithful_adv = (adv_predictions == clean_predictions)

        clean_acc = np.sum(is_correct_clean)/total_samples

        adv_acc = np.sum(is_correct_adv)/total_samples

        # Represents the fraction of samples that retain their clean prediction (correct or not) under attack
        adv_faithfulness = np.sum(is_faithful_adv)/total_samples

        # Represents the fraction of CORRECT samples that retain their clean prediction under attack
        adv_faithful_acc = np.sum( is_faithful_adv*is_correct_clean )/np.sum(is_correct_clean)

        results = {"clean_acc": clean_acc, "adv_acc": adv_acc, "adv_faithfulness": adv_faithfulness, "adv_faithful_acc": adv_faithful_acc}

        return results
    
    def _compute_sigma(self, mu: np.ndarray, mu2: np.ndarray):
        """Utility function for computing standard deviation"""
        sigma = np.sqrt(mu2 - mu**2).reshape(1,-1)
        return sigma
    
    def _compute_robustness_matrix(
                    self, 
                    mu_zi: np.ndarray, 
                    mu_z: np.ndarray, 
                    mu_i: np.ndarray, 
                    mu_z2: np.ndarray
                ) -> np.ndarray:
        
        """Computes robustness matrix from mean values
        
        Args:
            mu_zi (np.ndarray): E[Z*I]
            mu_z (np.ndarray): E[Z]
            mu_i (np.ndarray): E[I]
            mu_z2 (np.ndarray): E[Z^2]

        Returns:
            R (np.ndarray): robustness matrix with shape (n_classes, n_features)
        """

        # Compute standard deviation
        sigma_z = self._compute_sigma(mu_z, mu_z2)
        sigma_i = np.sqrt(mu_i.reshape(-1) - mu_i.reshape(-1)**2).reshape(-1,1)

        # Compute Pearson Correlation Coefficients
        R = (mu_zi - mu_z.reshape(1,-1) * mu_i)/(sigma_z * sigma_i + 1e-20)

        return R
    
    def compute_feature_robustness(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes feature-level robustness
        
        Returns:
            R_clean, R_adv (Tuple[np.ndarray, np.ndarray]): matrix of class/feature wise robustness of shape (nb_classes, nb_features)
        """

        total_samples = len(self.true_labels)

        mu_z_clean = self.mu_z_clean.get_value()
        mu_z_adv = self.mu_z_adv.get_value()

        mu_i = self.class_counts.reshape(-1,1)/total_samples

        mu_z2_clean = self.mu_z2_clean.get_value()
        mu_z2_adv = self.mu_z2_adv.get_value()

        mu_zi_clean = self.mu_iz_clean.get_value()
        mu_zi_adv = self.mu_iz_adv.get_value()

        R_clean = self._compute_robustness_matrix(
                                            mu_zi=mu_zi_clean,
                                            mu_z=mu_z_clean,
                                            mu_i=mu_i,
                                            mu_z2=mu_z2_clean
                                        )
        
        R_adv = self._compute_robustness_matrix(
                                            mu_zi=mu_zi_adv,
                                            mu_z=mu_z_adv,
                                            mu_i=mu_i,
                                            mu_z2=mu_z2_adv
                                        )

        R_clean = np.nan_to_num(R_clean, nan=0)
        R_adv = np.nan_to_num(R_adv, nan=0)

        return R_clean, R_adv

if __name__ == "__main__":

    import sys

    CONFIG_PATH = sys.argv[1]

    config = load_yaml_config(CONFIG_PATH) # Load config

    def main():

        # Make directory to save results
        results_path = config['results_path']
        os.mkdir(results_path)
        
        # Load data and model
        weights_name = config['weights_name']
        supervised_model_type = config.get('model_type', None)
        model_kwargs = config.get('model_kwargs', {})
        
        if supervised_model_type:
            model = SupervisedResNet50Model(model_type=supervised_model_type, weights_name=weights_name, **model_kwargs)
        else:
            model = SSLResNet50Model(weights_name, **model_kwargs)
        
        dataloader = get_imagenet_dataloader(**config['imagenet_kwargs'])

        # Load attack
        attack = ATTACK_INFO[config['attack']](
                        model = model, 
                        **config['attack_kwargs']
                    )
        # Tracker to maintain required records for robustness computation
        tracker = RobustnessTracker(model=model, nb_classes=1000, nb_features=2048)

        for x, y in tqdm.tqdm(dataloader):
            
            # More to GPU
            x = x.cuda()
            y = y.cuda()

            # Generate adversarial sample
            x_adv = attack(x)
            torch.cuda.empty_cache()

            tracker.update(x, x_adv, y)

        classifier_metrics = tracker.compute_classifier_robustness()
        R_clean, R_adv = tracker.compute_feature_robustness()

        # Save
        json.dump(config, open(os.path.join(results_path, 'config.json'), 'w'))
        json.dump(classifier_metrics, open(os.path.join(results_path, 'classifier_metrics.json'), 'w'))
        pickle.dump(R_clean, open(os.path.join(results_path, 'R_clean.pkl'), 'wb'))
        pickle.dump(R_adv, open(os.path.join(results_path, 'R_adv.pkl'), 'wb'))
        pickle.dump(tracker, open(os.path.join(results_path, 'tracker.pkl'), 'wb'))
    
    main()