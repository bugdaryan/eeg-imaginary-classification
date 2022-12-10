import sys
sys.path.append('../')
from inference import inference
import tensorflow as tf
import pytest
import os
import shutil


data_in_path = 'test_data/in_path'
data_out_path = 'test_data/out_path'
model_path = '../trained_models/bestModel_v2.h5'
results_path = 'test_data/results/'

def simple_inference():
    predictions, errors = inference(data_in_path, data_out_path, model_path, results_path)
    
    return predictions, errors

def delete_files():
    if os.path.isdir(data_out_path):
        shutil.rmtree(data_out_path)
    if os.path.isdir(results_path):
        shutil.rmtree(results_path)


def test_model_inference_no_errors():
    predictions, errors = simple_inference()
    
    assert errors == 0
    delete_files()

def test_model_inference_results_saved():
    predictions, errors = simple_inference()

    assert os.path.isfile(os.path.join(results_path, "predictions.npy")) == True
    delete_files()


def test_model_inference_two_predictions_are_same():
    predictions, errors = simple_inference()
    predictions2, errors2 = simple_inference()

    assert (predictions == predictions2).all() == True
    assert errors == 0
    assert errors2 == 0
    delete_files()