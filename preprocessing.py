import os
import numpy as np
from tqdm import tqdm
from data_utils import Utils
from constants import channels, exclude

def preprocess_from_dir(data_in_path, data_out_path):
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
    return subjects, errors