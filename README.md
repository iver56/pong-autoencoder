# Setup (Ubuntu)


## Set up nodejs with canvas support to generate a dataset

* Install nodejs & npm
* Install yarn
* `sudo apt-get install libcairo2-dev libjpeg8-dev libpango1.0-dev libgif-dev build-essential g++`
* `yarn`

## Set up python environment

* Install Anaconda with Python 3.6
* `conda install scipy==0.19.1 h5py==2.7.0 scikit-image==0.13.0`
* `pip install tensorflow==1.5.0 Keras==2.1.3`

# Usage

* Generate dataset: `node generateImages.js`
* Train: `python train.py`
* Run trained model on validation set: `python predict.py`
