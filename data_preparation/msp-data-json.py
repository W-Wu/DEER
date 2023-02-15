'''
Generating json file for training and testing.
'''

import os
import json
from speechbrain.dataio.dataio import read_audio
import numpy as np


def prepare_msp_podcast(
    data_folder,
    partition_dic,
    output_dir,
):

    for k,v in partition_dic.items():
        print(f"Creating {k}")
        fea_list = [os.path.join(data_folder,x) for x in v]
        create_json(fea_list, os.path.join(output_dir,k+'.json'))


def create_json(fea_list, json_file):
    json_dict = {}
    for fea_file in fea_list:
        signal = read_audio(fea_file)
        duration = signal.shape[0]
        seg_name = fea_file.split('/')[-1]

        # Create entry for this utterance
        json_dict[seg_name] = {
            "seg_name": seg_name,
            "fea_path": fea_file,
            "duration": duration,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    print(f"{json_file} successfully created!")


if __name__ == "__main__":
    wav_dir='/MSP-PODCAST-Publish-1.8/Audios_16k/'
    output_dir='msp-data/'
    label_dic = np.load('msp-data/msp-label.npy',allow_pickle=True).item()
    partition_dic = np.load('msp-data/partition_dic.npy',allow_pickle=True).item()
    prepare_msp_podcast(
    data_folder=wav_dir,
    partition_dic=partition_dic,
    output_dir = output_dir,
)
