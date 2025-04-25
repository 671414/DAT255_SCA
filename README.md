🔐 Deep Learning-Based Key Byte Recovery from Tiny-AES Encrypted Power Traces
This project is part of the DAT255 Deep Learning Engineering course at the Western Norway University of Applied Sciences (HVL).

Our goal is to recover a single key byte from Tiny-AES encrypted power traces using deep learning. We build upon existing models from SCAAML and train them on our own dataset, collected using ChipWhisperer Nano.

We’ve collected two datasets, each with 1,000,000 traces. As a fallback, we also support training on datasets from the SCAAML library.

📦 Project Structure
This project mainly consists of Jupyter notebooks, chosen for their ease of documentation, demonstration, and experimentation. Notebooks fall into three categories:

Core notebooks – Training, tuning, and testing models

Utility notebooks – Visualization, dataset processing, etc.

Baseline notebooks – Training and testing on the original SCAAML data

Some notebooks were developed for Google Colab, and might feel messy at first glance. This is due to time constraints during tuning/training, where a manual setup was “good enough” for our needs.

⚙️ Setup Notes
⚠️ TensorFlow version matters!
We’ve experienced import issues that were mostly resolved by importing from keras directly instead of tensorflow.keras.

⚠️ Issues with scaaml imports?
The easiest fix is to copy the function from the library directly into the notebook, instead of importing it.

📁 Dataset
We provide:

Raw datasets (from power trace capture)

Processed datasets (ready for training, validation, and testing)

Some notebooks expect raw traces, others need the pre-processed datasets (check their individual descriptions).

🚀 Getting Started
🔧 Training with our own data
Run in Google Colab:

keras_tuner_own_data.ipynb

Run hyperparameter search using training and test sets.

Lightly documented, relatively easy to follow.

train_model_own_data.ipynb

Trains a model using the best parameters.

Uses a json file with hyperparameters (e.g., stm32f0_tinyaes.json).

If running directly after the tuner, JSON might already be available in session.

🧠 Key Recovery on Our Data
Notebook: key_recovery_own_data.ipynb
This is the core of the project — test if our trained models can recover AES key bytes.

Based on SCAAML’s key recovery notebook, but adapted for our datasets.

Two key cells:

Evaluate the model using metrics

Attempt key byte recovery

Use datasets from the attack folder (processed and ready for this step)

📉 Baseline Models (SCAAML)
Want to recreate the SCAAML baseline?

We do not recommend this unless you're deeply interested. These were used only as a starting point.

🏗️ Training a Baseline
Notebook: notebooks/scaaml/train_scaaml_model.ipynb

Requires access to SCAAML’s datasets

Only needs the file sub_bytes_in0.npz

Adjust config values like max_trace_length before training

🔑 Key Recovery with Baseline
Notebook: notebooks/scaaml/key_recovery_with_changes.ipynb

Requires access to full SCAAML dataset

Based on original notebook, with minor changes

Link to data: SCAAML Intro Dataset

📚 Reference
bibtex
Kopier
Rediger
@inproceedings{burszteindc27,
  title={A Hacker Guide To Deep Learning Based Side Channel Attacks},
  author={Elie Bursztein and Jean-Michel Picod},
  booktitle ={DEF CON 27},
  howpublished = {\url{https://elie.net/talk/a-hackerguide-to-deep-learning-based-side-channel-attacks/}},
  year={2019},
  editor={DEF CON}
}
🙋‍♂️ Contributors
This project was completed as part of the group work in the DAT255 course.
