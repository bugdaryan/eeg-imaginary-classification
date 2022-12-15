import numpy as np
import os
import pandas as pd
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from experiments.models.model import EEGClassifier as EEGClassifierBase
from data_utils import Utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from livelossplot import PlotLossesKeras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import warnings 
import logging
import argparse
from preprocessing import preprocess_from_dir
from constants import channels, exclude

warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)

def inference(data_in_path, data_out_path, model_path, results_path):

    subjects, errors = preprocess_from_dir(data_in_path, data_out_path)
    
    x, _ = Utils.load(channels, subjects, base_path=data_out_path)
    reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    x_train_scaled_raw = minmax_scale(reshaped_x, axis=1)
    x_train = x_train_scaled_raw.reshape(x_train_scaled_raw.shape[0], int(x_train_scaled_raw.shape[1]/2), 2).astype(np.float64)

    model = EEGClassifierBase()
    loss = tf.keras.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=10e-4)

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model(np.random.rand(1, 640, 2))
    model.load_weights(model_path)
    predictions = model.predict(x_train)
    predictions = np.argmax(predictions, axis=1)

    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    np.save(os.path.join(results_path, "predictions.npy"), predictions, allow_pickle=True)

    return predictions, errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_in_path', type=str, default='../physionet.org/files/eegmmidb/1.0.0', help='path to data_in')
    parser.add_argument('-o', '--data_out_path', type=str, default='data_out/', help='path to data_out')
    parser.add_argument('-m', '--model_path', type=str, default='trained_models/bestModel_v2.h5', help='path to trained model')
    parser.add_argument('-r', '--results_path', type=str, default='results/', help='path to results')
    args = parser.parse_args()
    data_in_path = args.data_in_path
    data_out_path = args.data_out_path
    model_path = args.model_path
    results_path = args.results_path

    inference(data_in_path, data_out_path, model_path, results_path)[0]
