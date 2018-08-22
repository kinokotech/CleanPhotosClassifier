# CleanPhotosClassifier
- A goal of this project is to develop Android app which can clean up your photo folder.
- The sample android source code is published at [CleanPhotosAndroidSample](https://github.com/kinokotech/CleanPhotosAndroidSample).

## Require 
- bazel 0.15.1 and up
- Python 3.6.1
  - tensorflow 1.9.0rc1 
  - scikit-lean 0.19.1 
  - opencv-python 3.4.1.15 

## Setup
```
$ git clone git@github.com:tensorflow/tensorflow.git 
$ cd tensorflow/
$ bazel build tensorflow/python/tools:freeze_graph
$ bazel build tensorflow/contrib/lite/toco:toco 
```

## How to create model for TFlite
```
$ cd src/
$ python cnn.py
$ sh freeze_graph.sh
$ sh toco_grah.sh
```
The model for tflite will be generated :)   
