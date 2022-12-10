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


def train(data_in_path, data_out_path, model_checkpoints_path, epochs=100, batch_size=32, patience=10):
    if not os.path.isdir(data_out_path):
        os.mkdir(data_out_path)
    
    subs = os.listdir(data_in_path)
    subjects = [int(x[1:]) for x in subs]
    subjects = [s for s in subjects if s not in exclude]
    runs = [4, 6, 8, 10, 12, 14]

    for couple in channels:
        save_path = os.path.join(data_out_path, couple[0] + couple[1])
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        for sub in tqdm(subjects):
            x, y = Utils.epoch(Utils.select_channels
                (Utils.eeg_settings(Utils.del_annotations(Utils.concatenate_runs(
                Utils.load_data(subjects=[sub], runs=runs, data_path=data_in_path)))), couple),
                exclude_base=False)

            np.save(os.path.join(save_path, "x_sub_" + str(sub)), x, allow_pickle=True)
            np.save(os.path.join(save_path, "y_sub_" + str(sub)), y, allow_pickle=True)
    
    x, y = Utils.load(channels, subjects, base_path=data_out_path)
    y_one_hot  = Utils.to_one_hot(y, by_sub=False)
    reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(reshaped_x,
                                                                            y_one_hot,
                                                                            stratify=y_one_hot,
                                                                            test_size=0.20,
                                                                            random_state=42)

    x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
    x_test_valid_scaled_raw = minmax_scale(x_valid_test_raw, axis=1)

    x_valid_raw, x_test_raw, y_valid, y_test = train_test_split(x_test_valid_scaled_raw,
                                                        y_valid_test_raw,
                                                        stratify=y_valid_test_raw,
                                                        test_size=0.50,
                                                        random_state=42)

    x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], int(x_valid_raw.shape[1]/2),2).astype(np.float64)
    x_test = x_test_raw.reshape(x_test_raw.shape[0], int(x_test_raw.shape[1]/2),2).astype(np.float64)

    print('classes count')
    print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))

    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    x_train_smote_raw, y_train = sm.fit_resample(x_train_scaled_raw, y_train_raw)
    print('classes count')
    print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
    print ('after oversampling = {}'.format(y_train.sum(axis=0)))

    x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1]/2), 2).astype(np.float64)
    
    learning_rate = 1e-4
    if not os.path.isdir(model_checkpoints_path):
        os.mkdir(model_checkpoints_path)
    loss = tf.keras.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = EEGClassifierBase()
    modelPath = os.path.join(model_checkpoints_path, 'bestModel.h5')

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(
        model_checkpoints_path,
        monitor='val_acc',
        verbose=0, 
        save_best_only=True, 
        save_weights_only=False,
        mode='auto',
        save_freq=1 
    )

    earlystopping = EarlyStopping(
        monitor='val_acc', 
        min_delta=0.001, 
        patience=patience, 
        restore_best_weights=True, 
        verbose=0, 
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
        factor=0.5, patience=patience, 
        verbose=1, mode='auto', 
        min_lr=0.0000001
    )
    callbacksList = [checkpoint, earlystopping, reduce_lr, PlotLossesKeras()]

    hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_valid, y_valid), callbacks=callbacksList, verbose=1) 

    model.save_weights(model_checkpoints_path+'model_weights.h5')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_in_path', type=str, default='../physionet.org/files/eegmmidb/1.0.0', help='path to data_in')
    parser.add_argument('-o', '--data_out_path', type=str, default='data_out/', help='path to data_out')
    parser.add_argument('-m', '--model_checkpoints_path', type=str, default='model_checkpoints/', help='path to model_checkpoints')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('-p', '--patience', type=int, default=4, help='patience')
    args = parser.parse_args()
    data_in_path = args.data_in_path
    data_out_path = args.data_out_path
    model_checkpoints_path = args.model_checkpoints_path
    epochs = args.epochs
    batch_size = args.batch_size
    patience = args.patience

    train(data_in_path, data_out_path, model_checkpoints_path, epochs, batch_size, patience)
