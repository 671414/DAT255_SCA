import h5py as h5py
import tensorflow as tf
import numpy as np
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
        print(plaintext_set.shape, key_set.shape)

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
        key = np.transpose(key_set, (1, 0))
        text = np.transpose(plaintext_set, (1, 0))
        #Skriver ut de 3 første nøklene, greit for feilsøking
        print(key.shape, text.shape)
        print("org keys")
        print(key_set[0])
        print(key_set[256])
        print(key_set[512])
        print("transformed keys")
        print(key[:,0])
        print(key[:,256])
        print(key[:,512])
        print("transformed text")
        #her er 0 byte 0, 1 byte 1, 2 byte 2 osv..
        #printer alle radene i kolonne 0 som er byte 0 verdien i alle kolonnene
        print(text[:,0])
        print(text[:,1])
        print(text[:,2])



    # index to mark start and stop for slicing
    #set num increases if we have to split the dataset into smaller parts. NOT TESTED YET
    group_start_index = 0
    group_stop_index = trace_per_key

    dataset_name = f"{new_set_name}_{set_num}"
    f = h5py.File(f"{dataset_name}.hdf5", "w")

    # Loop trough the dataset, creating groups for every key
    for i in range(keys):
        group_name = key_set[group_start_index].tobytes().hex()
        # Create one group representing a shard
        group = f.create_group(group_name, track_order=True)

        group.create_dataset("traces", data=scaled_trace_set[group_start_index:group_stop_index, :, :])
        #her er det noe som feiler
        group.create_dataset("key", data=key[:, group_start_index:group_stop_index])
        print("text", text[group_start_index:group_stop_index])
        group.create_dataset("sub_bytes_in", data=sub_byte_in[:, group_start_index:group_stop_index])
        group.create_dataset("pts", data=text[:, group_start_index:group_stop_index])
        # group.create_dataset("sub_bytes_out", data = sub_byte_out[:, start_index:stop_index])
        #Siden alle radene er like, pga samme nøkkel blir det samme tallet repetert
        print(key[:, group_start_index:group_stop_index])
        group_start_index += trace_per_key
        group_stop_index += trace_per_key

    set_num += 1
    f.close()

#This part of the code is heavily inspired, and some parts borrowed from scaaml.
#We have done the necesary changes to make it work with our datasets
#referance will be provided.



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

#For our use casae i think returning a list with all the necessary information is better than a new dataset
#will contain data from the specified amount of traces, and all or specified amount of shard
#must be iterable so that shard 1 is stored i index 0etc, must be easy to understand
#må se på ferdigprossesert datasett, med sub bytes inn, eller gjøre det selv...
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
            #henter nøkkelbyten, men dette blir bare det et tall, fra 0 - 255
            k = group_name["key"][attack_byte, :num_traces]
            print(k)
            #henter tekstbyten, mens denne blir en lang liste på pts[i]
            pts = group_name["pts"][attack_byte][:num_traces]
            print(pts)

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
    print("done")
    return x_list, y_list, k_list, pts_list

def close_file(file_path: str):
    with h5py.File(file_path, "r") as f:
        f.close()
