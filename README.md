🔐 Deep Learning-Based Key Byte Recovery from Tiny-AES Encrypted Power Traces
This project was completed as part of the DAT255 Deep Learning Engineering course at the Western Norway University of Applied Sciences (HVL).

Our goal is to recover a single key byte from Tiny-AES encrypted power traces using deep learning. We build upon models from SCAAML and train them on our own datasets, collected using the ChipWhisperer Nano.

We collected datasets on two ChipWhisperers — one for training/testing data, and one for proper unseen holdout data. All necessary data, except full scaaml datasets are available here: https://huggingface.co/datasets/KasparER/DAT255_SCA/tree/main. 

📦 Project Structure
This project mainly consists of Jupyter notebooks, chosen for their ease of documentation, demonstration, and experimentation.

Notebooks fall into three categories:

Core notebooks – Training, tuning, and testing models

Utility notebooks – Visualization, dataset processing, etc.

Baseline notebooks – Training and testing on the original SCAAML data

Some notebooks were developed for Google Colab under time constraints, so setup might feel a bit manual/messy at first glance.

⚙️ Setup Notes
⚠️ TensorFlow version matters!
We experienced import issues, mostly resolved by importing from keras directly instead of tensorflow.keras.

⚠️ Issues with scaaml imports?
A quick fix is to copy the necessary functions directly into the notebook, instead of importing them.

📁 Dataset
We provide:

Raw datasets (640MB each) — captured power traces, found in the training/, testing/, and validation/ folders. Marked with raw.

Processed datasets (1.2GB each) — ready for training, validation, and testing, also located in the same folders, marked with the ChipWhisperer used.

Models and configurations:

Our models were trained on a STM32F0 target board.

v0 model: SCAAML hyperparameters (for STM32F415)

v1 model: Our own tuned hyperparameters

The important notebooks (tuning, training, and testing) expect the 1.2GB processed datasets.

🚀 Running the Project
🧪 Hyperparameter Tuning and Training
Run these notebooks on Google Colab:

keras_tuner_own_data.ipynb

training_on_our_data.ipynb

Both expect testing/testing_cw2.hdf5 and training/training_cw2.hdf5 to be available.

Notes:
The notebooks contain explanations of our model search process. Some JSON parsing code is present (from working in Colab), but can be ignored — we provide both the SCAAML and our best hyperparameter JSON files.

🔑 Key Recovery (Our Models)
Run on your local CPU.

Notebook: key_recovery_own_data.ipynb

This is the core notebook for validating if our trained models can recover AES key bytes.

Use datasets from the validation/ folder and pre-trained models from the Models/ folder.

Depending on your hardware and number of traces, this might take 10+ minutes.

Select:

The model you want to load

The number of traces to load

The dataset to validate on

The first evaluation cell runs recovery attempts across the dataset and collects metrics.
The final evaluation cell attempts key byte recovery — you can select which shard (key byte) to recover and how many traces to use.

📉 Baseline Models (SCAAML)
Want to recreate the original SCAAML baseline? (Not necessary unless you are deeply interested.)

🏗️ Training a Baseline Model
Run on Google Colab.

Notebook: notebooks/scaaml/train_scaaml_model.ipynb

Requires access to the SCAAML datasets (only needs sub_bytes_in0.npz).

Adjust config values like max_trace_length before training.

🔑 Key Recovery with Baseline Model
Run on local computer.

Notebook: notebooks/scaaml/key_recovery_with_changes.ipynb

Requires access to the full SCAAML dataset.

Based on the original SCAAML recovery notebook, with minor changes.

Dataset link: SCAAML Intro Dataset

📚 Reference
bibtex
Kopier
Rediger
@inproceedings{burszteindc27,
  title={A Hacker Guide To Deep Learning Based Side Channel Attacks},
  author={Elie Bursztein and Jean-Michel Picod},
  booktitle={DEF CON 27},
  howpublished={\url{https://elie.net/talk/a-hackerguide-to-deep-learning-based-side-channel-attacks/}},
  year={2019},
  editor={DEF CON}
}
🙋‍♂️ Contributors
This project was completed as part of a group project in the DAT255 Deep Learning Engineering course.
