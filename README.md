# ransomCare
A small proof-of-concept memory dump searching model to help users recover symmetric keys in memory.

## Introduction

This project is an extension of the work I've done the past two years with regard to ransomware research. My previous attempts to implement key extraction technology, while technically feasible, had major flaws. One of these flaws was the lack of an effective memory dump searching procedure to extract a ransomware key. While this was done with ransomwares that had known vulnerabilities such as WannaCry, there was no general implementation that could be applied generally to ransomware. This project is an attempt to address this flaw.

**Currently only Linux systems running Python 3 are supported.** I don't really want to deal with memory in Windows, I've had enough of that :D

I will probably write some sort of documentation or paper to actually combine all these years together and create a comprehensive ransomware defense. Stay tuned.

## Prerequisites
Your system should be running a GNU/Linux operating system, preferrably Ubuntu 22.04 LTS. Python 3+ should be installed.
If you have an NVIDIA graphics card to train the model, there is support for that.
I recommend using Jupyter Notebook to make life easier.
`gdb` is required to create the dataset.

Python Packages:
- os
- subprocess
- hashlib
- random
- string
- csv
- time
- sys
- cryptography
- torch
- sklearn
- pandas
- numpy
- tqdm

I may have forgotten something as these dependencies probably depend on other things.

## Files

`dataset.py` - Automatically generates a memory dump dataset. 
  Parameters:
  - `DATABASE_FILE`: the name of the dataset to export to, surround with double quotes and should be a `.csv` file.
  - `GDB_OUTPUT_FILE_PREFIX`: the prefix of the memory dumps exported by gdb
  - `NUM_CYCLES`: the number of dumps to take
  - `FILE_TYPES`: a list of different file extensions to be used at random for the test file to be encrypted

`CNN.py` - Creates and trains a CNN to identify AES keys in memory.
  Parameters:
  - `DATABASE_FILE`: the name of the dataset to import from, surround with double quotes and should be a `.csv` file.
  - `BATCH_SIZE`: size of batch
  - `LEARNING_RATE`: rate of learning
  - `MAX_EPOCHS`: epochs to run training for
  - `INPUT_LENGTH`: length of input, memory dumps can get quite large, input will always contain a key randomly spaced within the memory dump
  - `DEVICE`: GPU to accelerate training or just plain old CPU
  - `KEY_LENGTH`: Maximum key length of AES keys in the dataset, shorter keys will be padded and loss masking applied

`test.py` - Tests the efficacy of the model.
  Parameters:
  - `BATCH_SIZE`: size of batch
  - `LEARNING_RATE`: rate of learning
  - `DEVICE`: GPU to accelerate training or just plain old CPU
  - `MODEL_PATH`: file path to the model file
  - `TEST_DATA`: file path to the .csv dataset

`cleanup.sh` - Cleans up the environment by deleting dataset files such that a new dataset can be generated. GENERATED MODELS DO NOT GET DELETED.

## Dataset Notes

The dataset generated uses several techniques to try and add variation to the memory dumps.
- Every other memory dump generated is a synthetic dump of random byte arrays with a key mixed in.
- Random file types are encrypted, and the sizes of these files are randomized.
- Different simulated behaviors: partial encryption, key wrapping

## History

I started making automated key-extraction systems in fall of 2022, where I made a basic utility to detect ransomware attacks and automatically dump program memory by use of indicators such as high entropy file modification. This was the original PoC for an automated key-extraction system, and while it has no place on any consumer device or enterprise server, it demonstrated potential viability. In the fall of 2023, I began work on integrating machine learning into this system, using various classifiers to try and pinpoint ransomware programs more accurately than my previous PoC. It boasted better results (67% versus 97%) at the cost of being much less flexible in terms of deployment. Thanks to the clunkiness of Win32API.

Anyways that was incredibly difficult and I'm still working on getting that data on the internet. It's a couple hundred gigabytes of dataset, and even when minified its still too large to realistically host with my college student finances. So I decided to work on another improvable aspect that would be less tedious to make public as FOSS, and thus this was born out of winter break boredom.

Thank you for reading this lengthy thingy thing.

### Acknowlegements

I would like to thank my parents for letting me delay my meals by a few hours to work on this project, as well as work into the night. I also would like to thank my sister who has statistics background, her insight is brilliant. Thanks to my original mentor Dr. Tarek Saadawi for getting me interested in ransomware research, and my high school mentor Ari Basen for his immense help in educating me about new machine learning/AI technologies as well his guidance in this project. Also thanks to Gemini 2.0 Flash for cleaning up my code so that it actually is presentable on the internet.
