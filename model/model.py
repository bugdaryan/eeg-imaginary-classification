import os
from experiments.models.model import EEGClassifier as EEGClassifierBase
import tensorflow as tf
import numpy as np

class EEGClassifier:
    def __init__(self):
        self.model = None
        self.path = None
        self.weights_only = None
        self.model_loaded = False
    
    def load_model(self, path='experiments/model_checkpoints/bestModel_v2.h5', weights_only=True):
        if weights_only:
            if not os.path.isfile(path):
                raise FileNotFoundError(f'File {path} not found.') 
            model = EEGClassifierBase()
            loss = tf.keras.losses.categorical_crossentropy
            optimizer = tf.keras.optimizers.Adam(learning_rate=10e-4)

            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            model.built = True
            model(np.random.rand(1, 640, 2))
            model.load_weights(path)
        else:
            if not os.path.isdir(path):
                raise FileNotFoundError(f'Directory {path} not found.') 
            model = tf.keras.models.load_model(path)
        self.model = model
        self.path = path
        self.weights_only = weights_only
        self.model_loaded = True
        
    
    def preprocess_data(X):
        pass
    
    def preprocess_label(y):
        pass
    
    def predict(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess_data(X)
        y_pred = self.model.predict(X)
        
        return y_pred
    
    def evaluate(X, y, preprocess=True):
        if preprocess:
            X = self.preprocess_data(X)
            y = self.preprocess_label(y)
        loss, acc = self.model.evaluate(X, y)
        
        return loss, acc