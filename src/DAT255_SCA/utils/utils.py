dataset_name = "attack"
shard = []
attack_byte = 0
attack_point = "sub_bytes_in"
max_trace_length = 5000
num_traces = 256
num_traces_per_shard = 256
full_key = False
x_List: List[Tensor] = []
y_List: List[Tensor] = []

#need to open the file to read any info
with (h5py.File(f"{dataset_name}.hdf5", "r")) as f:
    for group in f.keys():
        group_name = f[group]

        x_shard = group_name["traces"][:num_traces_per_shard, :5000, :]
        x_shard = tf.convert_to_tensor(x_shard, dtype="float32")

        y_shard = group_name["sub_bytes_in"][attack_byte]
        y_shard = y_shard[:num_traces_per_shard]
        y_shard = tf.keras.utils.to_categorical(y_shard, 256)
        y_shard = tf.convert_to_tensor(y_shard, dtype="uint8")

        k = group_name["key"][attack_byte][:num_traces_per_shard]
        pts = group_name["pts"][attack_byte][:num_traces_per_shard]

