# eeg-imaginary-classification

### Prerequisites
- Ensure that Python 3.7 or above is installed on your computer

### Steps
1. Clone the repository from the branch that you want to have locally installed.
2. Create virtual envoronment
3. Locate the clonned repository in your computer and install all package requirements by running `pip install -r requirements.txt` in the terminal located in the repository directory


### Data
  - Dataset source: https://physionet.org/content/eegmmidb/1.0.0/
 
 In order to get the dataset, we will use wget to download dataset to current directory

- ```!wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/```

This will create a folder named `physionet.org` folder with all the eeg data in it.

### EDA

Check [experiments/EDA.ipynb](https://github.com/bugdaryan/eeg-imaginary-classification/blob/master/experiments/EDA.ipynb) for more info for data.

### Training
```commandline
usage: train.py [-h] [--data_in_path] [--data_out_path] [--model_checkpoints_path] [--epochs] [--batch_size] [--patience]

optional arguments:
  -h, --help                    show this help message and exit
  --data_in_path                input data path
  --model_checkpoints_path      path to model_checkpoints
  --epochs                      number of epochs
  --batch_size                  batch size
  --patience                    patience
```

### Inference
```commandline
usage: inference.py [-h] [--data_in_path] [--data_out_path] [--model_path] [--results_path]

optional arguments:
  -h, --help                    show this help message and exit
  --data_in_path                input data path
  --model_path                  path to trained model
  --results_path                path to results
```
