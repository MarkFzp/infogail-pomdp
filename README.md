# Multi-Modal Imitation Learning in Partially Observable Environments

This repository contains data and TensorFlow 1.1x code for the preprint “[Multi-Modal Imitation Learning in Partially Observable Environments](https://markfzp.github.io/data/AAMAS2020_Imitation.pdf)".

# Dependency on Linux

1. Install cuda 10.0 if it's not available already. 

2. Install anaconda if it's not available already, and create a new environment. You need to install a few things, namely, OpenMPI, TensorFlow 1.12, Stable Baselines, OpenAI Gym and MuJoCo. (Please refer to "[this link](https://www.roboti.us/license.html)" for installation of MuJoCo physics simulator.)

```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev

wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
conda update -n base -c defaults conda
conda create --name mmim python=3.6
source activate mmim

conda install numpy pyyaml setuptools cmake cffi tqdm pyyaml scipy ipython mkl mkl-include cython typing h5py nltk spacy numpydoc scikit-learn jpeg

pip3 install mujoco-py
pip3 install stable-baselines[mpi]

conda install tensorflow-gpu=1.12.0
conda install gym
```

# Export Expert Demos

1. Import the Python files of customized partial observable Gym environments `mujoco/expert/hopper_v3.py` to the corresponding folder of the Gym local directory `[Gym dir]/gym/envs/mujoco/`.

2. Configurate `mujoco/expert/config.py` for desirable behaviors (default to train PPO expert from scratch for 5M iterations). 

3. Run `expert.py` to get `.npy` file containing expert demonstrations.

```
cd mojoco/expert
python3 expert.py
```

# Run Imitation

All the tunable hyperparameters and network structures including hidden dimensions, activation functions, learning rates, training iterations, gamma and lambda used in GAE, clipping range in PPO used for training the imitation policy are variables in `mujoco/config.py`.

```
cd mojoco/expert
python3 train.py
```

Training logs are printed to `stdout` if `config.log_path` is `None`. 






