Deep Learning-Based Key Byte Recovery from Tiny-AES Encrypted Power Traces





This project is a part of our DAT255 Deep learning engineering course at the Western Norway University Of Applied science.
Our task is to build upon already existing deep learning models, and train it on our own dataset.

Our project recovering a key-byte from a tiny-AES encrypted power trace.

We will use our own data, that we collect using chipwhisperer Nano. We have already gathered two datasets with 1 000 000 traces.
Our contingency plan is to use datasets from this library https://github.com/google/scaaml/blob/main/scaaml_intro/README.md

We are planning to use Resnet v2

Troughout this project we utilize the scaaml library directly, and trough modification and borrowing concepts. We will refer to where in the library, and when we used the specific parts.
@inproceedings{burszteindc27,
title={A Hacker Guide To Deep Learning Based Side Channel Attacks},
author={Elie Bursztein and Jean-Michel Picod},
booktitle ={DEF CON 27},
howpublished = {\url{https://elie.net/talk/a-hackerguide-to-deep-learning-based-side-channel-attacks/}}
year={2019},
editor={DEF CON}
}

How to recreate our results: 

Project structure:

Our project mostly consist of multiple jupyter notebooks. These provide easy documentation and demo ability and are suitable to demonstrate our project.

First you need datasets, we will provide training, testing and validation data. These will be in their first raw format, and also in their ready to train format.

Recreating baseline model:

Recreating our model:

Use keras tuner and training in colab...


