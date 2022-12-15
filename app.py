from flask import Flask, request
from inference import inference
import zipfile
import mne
import os 
import shutil

app = Flask(__name__)

data_in = 'data_in/'
data_out = 'data_out/'

@app.route('/predict', methods=['POST'])
def predict():
    if os.path.isdir(data_in):
        shutil.rmtree(data_in)
    os.mkdir(data_in)

    file = request.files['edf_files']  
    file_like_object = file.stream._file  
    with zipfile.ZipFile(file_like_object, 'r') as zip_file:
        zip_file.extractall(path=data_in)

    predictions, errors = inference(data_in, data_out, 'trained_models/bestModel_v2.h5', 'results/')
    predictions = predictions.argmax(axis=1)
    predictions = predictions.tolist()

    return {'predictions': predictions}


if __name__ == '__main__':
    app.run(debug=True)
