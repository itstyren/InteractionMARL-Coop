# InteractionMARL-Coop
Welcome to the InteractionMARL-Coop repository, the official codebase supporting the paper ["Enhancing Cooperation through Selective Interaction and Long-term Experiences in Multi-Agent Reinforcement Learning"](https://arxiv.org/abs/2405.02654). This repository offers a comprehensive training framework for fostering cooperative behaviors in multi-agent systems.

We provide a detailed description of the hyperparameters and training procedures in the Supplementary Material of our paper. We would strongly suggest to review these details thoroughly before executing the code to ensure optimal results.

# 1. Repository Structure
* `envs`: Contains environment implementations for spatial PGGs and wrappers for RL and EGT scenarios.
* `runner`: Includes code for training and evaluation processes.
* `scripts`: Features executable scripts for training with pre-set hyperparameters.
* `algorithms`:  the DQN algorithm code, adapted from [SB3](https://github.com/hill-a/stable-baselines).
* `configs`:  Stores configuration files detailing hyperparameters and environmental settings.

# 2. Installation
This framework is developed using PyTorch and supports both CPU and GPU environments. Below is an installation guide for a system with `CUDA==11.8` and `Python==3.10.0`, though other CUDA versions are also compatible.  For additional CUDA version support, please refer to the [Pytorch official website](https://pytorch.org/get-started/locally/).

Note: The framework has been tested exclusively in a Linux environment.

```bash
# Create a Python environment
conda create -n idsd python==3.10.0
conda activate idsd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```bash
# Install the InteractionMARL-Coop package
cd interaction-dynamics-in-social-dilemmas
pip install -e .
```

# 3. Training and Evaluation
Example using `train_lattirc_rl.sh`:
```bash
cd scripts/train_pd_script
chmod +x ./train_lattirc_rl.sh
./train_lattirc_rl.sh
```
Note: We utilize Weights & Biases for visualization. Please ensure you are registered and logged in on the Weights & Biases platform prior to training.

We additionally provide `./eval_lattice_rl.sh` for evaluation. 


# 4. Visualization
The `visualization` directory contains the Jupyter notebook `plot_result.ipynb`. This notebook compiles and visualizes the results reported in both the main paper and supplementary material.

# 5. Paper citation
Coming soon.
