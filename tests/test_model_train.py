import sys
sys.path.append('../')
from train import train
import tensorflow as tf
import pytest
import os
import shutil


data_in_path = 'test_data/in_path'
data_out_path = 'test_data/out_path'
model_checkpoints_path = 'test_data/model_checkpoints/'
epochs = 1
batch_size = 32
patience = 10

channels = [["FC1", "FC2"],
            ["FC3", "FC4"],
            ["FC5", "FC6"],
            ["C5", "C6"],
            ["C3", "C4"],
            ["C1", "C2"],
            ["CP1", "CP2"],
            ["CP3", "CP4"],
            ["CP5", "CP6"]]

def simple_train():
    model, errors = train(data_in_path, data_out_path, model_checkpoints_path, epochs, batch_size, patience)
    
    return model, errors

def delete_files():
    if os.path.isdir(data_out_path):
        shutil.rmtree(data_out_path)
    if os.path.isdir(model_checkpoints_path):
        shutil.rmtree(model_checkpoints_path)


def test_model_trained_no_errors():
    model, errors = simple_train()
    
    assert errors == 0
    assert isinstance(model, tf.keras.Model) == True
    assert model.built == True
    delete_files()

def test_model_trained_data_saved():
    model, errors = simple_train()
    subs = os.listdir(data_in_path)
    subjects = [int(x[1:]) for x in subs]
    
    for couple in channels:
        save_path = os.path.join(data_out_path, couple[0] + couple[1])
        for sub in subjects:
            assert os.path.isfile(os.path.join(save_path, "x_sub_" + str(sub) + ".npy")) == True
            assert os.path.isfile(os.path.join(save_path, "y_sub_" + str(sub) + ".npy")) == True
    
    delete_files()

def test_model_trained_model_saved():
    model, errors = simple_train()
    
    assert os.path.isfile(os.path.join(model_checkpoints_path, "model_weights.h5")) == True
    delete_files()