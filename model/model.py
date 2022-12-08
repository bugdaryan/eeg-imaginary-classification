import os
from experiments.models.model import EEGClassifier as EEGClassifierBase

class EEGClassifier:
    def __init__(self):
        pass
    
    
    def load_model(self, path='experiments/model_checkpoints/bestModel_v2.h5', weights_only=True):
        pass