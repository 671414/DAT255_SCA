import argparse
import json
import sys
from termcolor import cprint
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow.keras.backend as K

from scaaml.intro.model import get_model
from scaaml.utils import get_model_stub
from scaaml.utils import get_num_gpu
from scaaml.utils import tf_cap_memory
from scaaml.intro.generator import create_dataset

"""This is a version of the scaaml.intro.generator.create_dataset function that saves the
 dataset as a .npz file. We needed to change it a bit, and make it possible to run locally due to
 errors encounter with the create dataset method in google colab. We only ran this code long enough to 
 generate sub_bytes_in for byte 0 and the npz files are around 5GB each.
"""

def save_datasets(config):
    tf_cap_memory()
    algorithm = config["algorithm"]
    train_glob = f"datasets/{algorithm}/train/*"
    test_glob = f"datasets/{algorithm}/test/*"
    test_shards = 256
    num_traces_per_test_shards = 16
    batch_size = config["batch_size"] * get_num_gpu()
    print(get_num_gpu())
    for attack_byte in config["attack_bytes"]:
        for attack_point in config["attack_points"]:

            x_train, y_train = create_dataset(
                train_glob,
                batch_size=batch_size,
                attack_point=attack_point,
                attack_byte=attack_byte,
                num_shards=config["num_shards"],
                num_traces_per_shard=config["num_traces_per_shard"],
                max_trace_length=config["max_trace_len"],
                is_training=True)

            x_test, y_test = create_dataset(
                test_glob,
                batch_size=batch_size,
                attack_point=attack_point,
                attack_byte=attack_byte,
                num_shards=test_shards,
                num_traces_per_shard=num_traces_per_test_shards,
                max_trace_length=config["max_trace_len"],
                is_training=False)
            file_name=attack_point + attack_byte
            np.savez(file_name+'.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            print("Files saved")


config = """{
    "model": "cnn",
    "device": "stm32f415",
    "algorithm": "tinyaes",
    "version": "10",
    "attack_points": [
        "sub_bytes_out",
        "sub_bytes_in",
        "key"
    ],
    "attack_bytes": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15"
    ],
    "max_trace_len": 20000,
    "num_shards": 256,
    "num_traces_per_shard": 256,
    "batch_size": 32,
    "epochs": 30,
    "optimizer_parameters": {
        "lr": 0.001,
        "multi_gpu_lr": 0.001
    },
    "model_parameters": {
        "activation": "relu",
        "initial_filters": 8,
        "initial_pool_size": 4,
        "block_kernel_size": 3,
        "blocks_stack1": 3,
        "blocks_stack2": 4,
        "blocks_stack3": 4,
        "blocks_stack4": 3,
        "dense_dropout": 0.1
    }
}"""

if __name__ == "__main__":
    save_datasets(json.loads(config))