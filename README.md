# **Traffic Sign Classifier**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This repo contains the code written to complete the project **Traffic Sign Classifier** on Udacity Self-Driving Car Nanodegree. The goal is classify the most common traffic sign from images.

Prerequisites
---
To run this project, you need [Anaconda 4.3.30](https://anaconda.org/conda-canary/conda/files?version=4.3.30) installed.

Installation
---
First, clone the repository:
```
git clone https://github.com/shohne/CarND-Traffic-Sign-Classifier-Project.git
```
Change current directory:
```
cd CarND-Traffic-Sign-Classifier-Project
```
Create a conda environment with all dependencies:
```
conda env create -f environment.yaml
```
The name of created environment is *carnd-traffic-sign*.

Running the Notebook
---
Activate the created conda environment:
```
source activate carnd-traffic-sign
```
And run Jupyter Notebook:
```
jupyter notebook Traffic_Sign_Classifier.ipynb
```
Implementation Details
---
This notebook comes with a pre-trained model located in **model** directory. To train model from scratch, adjust variable **force_train** to value **True**. It is located in the block *Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas*. The training procedure assumes that all 3 (train.p, valid.p and test.p) dataset files are in **data** directory.

Please visit the [report.md](report.md) for more information about the model, training procedure and results.

List of Files
---
* **Traffic_Sign_Classifier.ipynb** main file with the code to build, train and evaluate the model to classify images;
* **model_traffic.png** image with neural network model implemented;
* **signames.cvs** contains description for each traffic sign class;
* **data** (directory) must contain dataset files *train.p*, *valid.p* and *test.p*;
* **webdata** (directory) contains sign images captured from web. These images are used to estimate model performance in *real* data;
* **model** (directory) contains pre-trained *Tensorflow* model;
* **README.md**
* [**report.md**](report.md)
* **environment.yaml** python dependencies.
