import h5py as h5py
import tensorflow as tf
import numpy as np


"""This module provides functionalities for creating, managing, inspecting, and processing HDF5 datasets, specifically designed to handle datasets
 containing cryptographic traces, keys, plaintexts, and processed attack points. It uses libraries like `h5py` for HDF5 file manipulation,
  `tensorflow` for data scaling and tensor conversion, and `numpy` for numerical operations.
"""

"""Creates and writes a new HDF5 dataset based on raw cryptographic trace data, plaintexts, and keys. For each unique key,
 a group is created, and the corresponding scaled trace data, processed attack points, plaintexts, and keys are stored.
 
**Parameters:**
- `file_path` (str): Path to the input HDF5 file containing the raw data.
- `keys` (int): Number of unique cryptographic keys.
- `trace_per_key` (int): Number of traces recorded per key.
- `new_set_name` (str): Name prefix for the generated dataset files.
"""
def create_dataset(file_path: str, keys: int, trace_per_key: int, new_set_name: str ):
    #creating a file in the filepath where the set is stored, with unique name
    file_path = file_path
    keys = keys
    trace_per_key = trace_per_key
    shard_size = trace_per_key*keys

    #one group for each key
    set_num = 0


    start_index = 0
    stop_index = (set_num+1)*shard_size
    #Dependent on knowing the filenames in the h5py file
    with h5py.File(file_path, "r") as f:
        plaintext_set = f['data'][start_index:stop_index, ]
        trace_set = f['trace'][start_index:stop_index, ]
        key_set = f['key'][start_index:stop_index, ]

        #scaling the trace, might want to move this to right before model training
        scaled_trace_set = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)(trace_set)
        scaled_trace_set = tf.expand_dims(scaled_trace_set, axis=-1)

        #precompute the attack point
        num_traces = len(trace_set)
        num_bytes = 16
        sub_byte_in_set = np.zeros((num_traces, 16), dtype=np.uint8)

        for i in range(num_traces):
            for byte_index in range(num_bytes):
                pts = plaintext_set[i][byte_index] ^ key_set[i, byte_index]
                sub_byte_in_set[i, byte_index] = pts

        sub_byte_in_set = sub_byte_in_set

        #transpose the sets into [byte][data]
        sub_byte_in = np.transpose(sub_byte_in_set, (1, 0))
        keyt = np.transpose(key_set, (1, 0))
        text = np.transpose(plaintext_set, (1, 0))


    f.close()
    # index to mark start and stop for slicing
    #set num increases if we have to split the dataset into smaller parts. NOT TESTED YET
    group_start_index = 0
    group_stop_index = trace_per_key
    dataset_name = f"{new_set_name}_{set_num}"
    f = h5py.File(f"{dataset_name}.hdf5", "w")
    #hvordan ser key ut siden :, 0 blir 0...255
    # Loop trough the dataset, creating groups for every key
    for i in range(keys):
        #group_name = key_set[group_start_index].tobytes().hex()
        group_name = f"{new_set_name}_{i}"
        # Create one group representing a shard
        group = f.create_group(group_name, track_order=True)

        group.create_dataset("traces", data=scaled_trace_set[group_start_index:group_stop_index, :, :])
        #her er det noe som feiler
        #forsøker å lagre nøkkelen en gang
        key = group.create_dataset("key", data=keyt[:, group_start_index:group_stop_index])
        #print("text", text[:,group_start_index:group_stop_index])
        group.create_dataset("sub_bytes_in", data=sub_byte_in[:, group_start_index:group_stop_index])
        group.create_dataset("pts", data=text[:, group_start_index:group_stop_index])
        # group.create_dataset("sub_bytes_out", data = sub_byte_out[:, start_index:stop_index])
        #Siden alle radene er like, pga samme nøkkel blir det samme tallet repetert
        #print(key[:, 0])
        group_start_index += trace_per_key
        group_stop_index += trace_per_key

    set_num += 1
    f.close()




"""THIS MEHTOD HAS NOT BEEN USED OR TESTED PROPERLY. Loads and processes trace and label data from an HDF5 dataset to prepare it for training machine learning models.

**Parameters:**
- `filepath` (str): Path to the HDF5 dataset file.
- `attack_point` (str): Type of attack point to target (`sub_bytes_in`, `key`, etc.).
- `attack_byte` (int): Byte index being used for the attack.
- `num_traces` (int): Total number of traces to process.
- `num_traces_per_key` (int): Number of traces extracted per key.
- `shuffle` (bool): Whether to shuffle the dataset. Defaults to `False`

**Return Values:**
- `x_train`: Tensor containing scaled traces for model training.
- `y_train`: Tensor containing one-hot encoded labels.

**Functionality:**
- Reads trace data and attack points from the HDF5 file.
- Converts traces into TensorFlow tensors.
- Encodes target labels using one-hot encoding with 256 categories (byte values).
- Returns the training features and labels for use in neural network training.

"""
def load_and_prepare_dataset_for_training(filepath: str, attack_point: str, attack_byte, num_traces, num_traces_per_key, shuffle: bool = False):
    dataset_name = filepath
    attack_point = attack_point
    attack_byte = attack_byte
    num_traces_per_shard = num_traces_per_key
    num_traces = num_traces #possible to default these as well
    #trace length is 5000 as this is the most we can get from chipwhisperer nano

    #shuffling kan muligens gjøres her, men hva med skalering? er vel best å gjøre begge på disse små datasettene istedet.

    with (h5py.File(f"{dataset_name}", "r")) as f:
        for group in f.keys():
            group_name = f[group]

            x_shard = group_name["traces"][:num_traces_per_shard, :5000, :]
            x_shard = tf.convert_to_tensor(x_shard, dtype="float32")

            y_shard = group_name["sub_bytes_in"][attack_byte]
            y_shard = y_shard[:num_traces_per_shard]
            y_shard = tf.keras.utils.to_categorical(y_shard, 256)
            y_shard = tf.convert_to_tensor(y_shard, dtype="uint8")

            x_List.append(x_shard)
            y_List.append(y_shard)
        x: Tensor = tf.concat(x_List, axis=0)
        y: Tensor = tf.concat(y_List, axis=0)
        #possible to shuffle these
    x_train = x
    y_train = y
    f.close()
    return x_train, y_train


"""Prepares a dataset for evaluation by extracting the relevant traces, keys, plaintexts, and labels for specified attack points.

**Parameters:**
- `filepath` (str): Path to the HDF5 dataset file.
- `attack_byte` (int): Byte index being targeted in the evaluation.
- `attack_point` (str): Attack point to evaluate (`sub_bytes_in`, `sub_bytes_out`, or `key`).
- `num_traces` (int): Total number of traces to process.

**Return Values:**
- `x_list`: List of feature tensors (scaled traces).
- `y_list`: List of one-hot encoded label tensors.
- `k_list`: List of key values for each trace group.
- `pts_list`: List of plaintext data for each trace group.

**Functionality:**
- Reads all groups in the dataset and extracts traces, attack points (labels), and other relevant data.
- Converts traces and labels into tensors for evaluation.
- Supports multiple groups in the file and organizes data into separate lists for easy use.
"""

def load_and_prepare_dataset_for_evaluation(filepath: str, attack_byte, attack_point, num_traces):
    dataset_name = filepath
    
    #we only have one attack_point, but it can be nice for future use
    #the list that will be returned
    k_list = []
    pts_list = []
    x_list = []
    y_list = []

    with (h5py.File(f"{dataset_name}", "r")) as f:
        for group in f.keys():
            group_name = f[group]


            #endret slik at nøkkelen bare lagres en gang
            k = group_name["key"][attack_byte][:num_traces]

            #print(k)
            #henter tekstbyten, mens denne blir en lang liste på pts[i]
            pts = group_name["pts"][attack_byte][:num_traces]
            #print(pts)

            x = group_name["traces"][:num_traces, :5000, :]
            x = tf.convert_to_tensor(x, dtype="float32")

            # load y
            if attack_point == "key":
                y = group_name["keys"][attack_byte]
            elif attack_point == "sub_bytes_in":
                y = group_name["sub_bytes_in"][attack_byte]
            elif attack_point == "sub_bytes_out":
                y = group_name["sub_bytes_out"][attack_byte]
            else:
                raise ValueError(f"Unknown attack point {attack_point}.")

            y = group_name["sub_bytes_in"][attack_byte]
            y = y[:num_traces]
            y = tf.keras.utils.to_categorical(y, 256)
            y = tf.convert_to_tensor(y, dtype="uint8")


            k_list.append(k)
            pts_list.append(pts)
            x_list.append(x)
            y_list.append(y)
        f.close()
    return x_list, y_list, k_list, pts_list

def close_file(file_path: str):
    with h5py.File(file_path, "r") as f:
        f.close()

"""def inspect_dataset(file_path: str):
    with h5py.File(file_path, "r") as f:
        i = 0
        for group in f.keys():
            group_name = f[group]
            print(group)
            i += 1
            print(i)
            print()
            for dset in f[group].keys():
                print(dset)
            keys = group_name["key"][0][:]
            print(keys)
            print()
        f.close()"""

"""Provides a deeper inspection of an HDF5 file, printing metadata for all datasets, including their shapes, data types, and values.

**Parameters:**
- `filepath` (str): Path to the HDF5 file for inspection.

**Functionality:**
- Iterates over all groups and datasets in the file.
- Prints detailed information, including dataset dimensions and raw data values.
"""
def new_inspect(filepath: str):
    data = h5py.File(filepath, 'r')
    for group in data.keys():
        print(group)
        for dset in data[group].keys():
            print(dset)
            ds_data = data[group][dset]  # returns HDF5 dataset object
            print(ds_data)
            print(ds_data.shape, ds_data.dtype)
            arr = data[group][dset][:]  # adding [:] returns a numpy array
            print(arr.shape, arr.dtype)
            print(arr)

"""Compares the keys stored in two different HDF5 files and identifies duplicate keys between them.

**Parameters:**
- `filepath1` (str): Path to the first HDF5 dataset file.
- `filepath2` (str): Path to the second HDF5 dataset file.

**Return Values:**
- A list of keys that are common in both files.

**Functionality:**
- Extracts keys from all groups in both files.
- Uses NumPy to compare keys for equivalence and identifies duplicates.
- Returns the list of duplicate keys, if found.
"""
def look_for_duplicate_keys(filepath1: str, filepath2: str):
    def extract_keys(filepath):
        keys = []
        with h5py.File(filepath, 'r') as data:
            for group in data.keys():
                group_name = data[group]
                if "key" in group_name:  # Sjekk at "key" finnes
                    keys.extend(group_name["key"][:, 0])  # Forutsetter at "key" er 2D
        return keys

    # Ekstraher nøkler fra begge filer
    keys1 = extract_keys(filepath1)
    keys2 = extract_keys(filepath2)

    # Finn overlappende nøkler (sett-operasjon for rask sammenligning)
    common_keys = list(set(keys1) & set(keys2))

    return common_keys

"""
Compares the keys stored in two different HDF5 files and identifies duplicate keys between them.

**Parameters:**
- `data1` (str): Path to the first HDF5 dataset file.
- `data2` (str): Path to the second HDF5 dataset file.

**Return Values:**
- A list of keys that are common in both files.

**Functionality:**
- Opens both HDF5 files using `h5py`.
- Iterates through all groups in the datasets to extract keys.
- Keys are extracted as 16-byte values (assumed to be stored in a "key" dataset).
- Converts key collections into sets and performs a set intersection to find duplicates.
- Prints the number of common keys found and each duplicate key.
- Returns a list of common keys.

**Example:**
```python
dup_keys = find_duplicate_keys("file1.h5", "file2.h5")
print(dup_keys)
```
"""

def find_duplicate_keys(data1, data2):
    with h5py.File(data1, 'r') as data1, h5py.File(data2, 'r') as data2:

        keys1 = []
        keys2 = []

        # Extract keys from data1
        for group in data1.keys():
            group_name = data1[group]
            keys = group_name["key"][:, 0]  # Extract 16-byte keys
            keys1.append(tuple(keys))  # Convert to tuple for hashing

        # Extract keys from data2
        for group in data2.keys():
            group_name = data2[group]
            keys = group_name["key"][:, 0]  # Extract 16-byte keys
            keys2.append(tuple(keys))  # Convert to tuple for hashing

        # Convert to sets for comparison
        set_keys1 = set(keys1)
        set_keys2 = set(keys2)

        # Find common keys using set intersection
        common_keys = set_keys1 & set_keys2

        if common_keys:
            print(f"Common keys found: {len(common_keys)}")
            for key in common_keys:
                print(key)
        else:
            print("No common keys found.")

        return list(common_keys)
