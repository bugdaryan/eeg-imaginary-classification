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
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)

channels = [["FC1", "FC2"],
            ["FC3", "FC4"],
            ["FC5", "FC6"],
            ["C5", "C6"],
            ["C3", "C4"],
            ["C1", "C2"],
            ["CP1", "CP2"],
            ["CP3", "CP4"],
            ["CP5", "CP6"]]
exclude = [38, 88, 89, 92, 100, 104]

def inference(data_in_path, data_out_path, model_path, results_path):
    if not os.path.isdir(data_out_path):
        os.mkdir(data_out_path)

    errors = 0
    
    subs = os.listdir(data_in_path)
    subjects = [int(x[1:]) for x in subs]
    subjects = [s for s in subjects if s not in exclude]
    runs = [4, 6, 8, 10, 12, 14]

    for couple in channels:
        save_path = os.path.join(data_out_path, couple[0] + couple[1])
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        for sub in tqdm(subjects):
            try:
                x, y = Utils.epoch(Utils.select_channels
                    (Utils.eeg_settings(Utils.del_annotations(Utils.concatenate_runs(
                    Utils.load_data(subjects=[sub], runs=runs, data_path=data_in_path)))), couple),
                    exclude_base=False)

                np.save(os.path.join(save_path, "x_sub_" + str(sub)), x, allow_pickle=True)
                np.save(os.path.join(save_path, "y_sub_" + str(sub)), y, allow_pickle=True)
            except:
                errors += 1
                continue
    
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
