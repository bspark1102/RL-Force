# Imitation-assisted Reinforcement Learning for Dynamic Passive System Control

## Overview
This repository contains code and resources related to the paper "Performance, Robustness, and Portability of Imitation-assisted Reinforcement Learning for Dynamic Passive System Control". The goal of this research is to implement policy-gradient reinforcement learning to optimize the control of building ventilation and insulation systems dynamically.

## Structure
The repository consists of Python and MATLAB scripts, EnergyPlus input files, and utility functions necessary to replicate and extend experiments detailed in the associated research paper.

### Necessary Files

The following files are essential for running the simulations:

- **Policy Gradient Agents:**
  - These files are under the folder './Python_utils'. Make sure they are under the same directory as `Main_Python.ipynb` when runnning.
  - `policygradient_tf3_dualI.py` & `policygradient_tf3_dualV.py`: Specialized dual agents for insulation and ventilation respectively.

- **MATLAB Simulation Interface:**
  - `Main_Matlab.m`: MATLAB-EnergyPlus co-simulation setup.

- **Main Python Script:**
  - `Main_Python.ipynb`: Main Jupyter notebook for initializing and training reinforcement learning agents.

- **Utilities:**
  - These files are under the folder './Matlab_utils'. Make sure they are under the same directory as `Main_Matlab.m` when runnning.
  - `pushBack.m`: MATLAB helper functions for data manipulation.
  - `normalizer.m`: Data normalization functions.
  - `get_reward.m`: Data normalization functions.
  - `select_action.m`: Function to .

- **EnergyPlus Models and Weather Files:**
  - These files are under the folder './Weather_and_IDF_files'. Make sure they are under the same directory as `Main_Matlab.m` when runnning.
  - `.idf` files define simulation models used by EnergyPlus (`070121j_Single_NoSunspace...`).
  - `.epw` files define the weather files used by EnergyPlus (`USA_NY_Albany.Intl....`).

- **Note**
  - The example provided here is designed for _further training_ of the pretrained networks, through reinforcement learning.

## Requirements

- **Python Libraries:**
  - TensorFlow (1.4)
  - NumPy
  - SciPy
  - os
  - random
  - socket
  - shutil
  - Jupyter Notebook
  - Python 3.x

- **MATLAB:**
  - MATLAB with Simulink support. (2019b for best compatibility)
  - MATLAB EnergyPlus (MLEP) Toolbox.

- **Simulation Software:**
  - EnergyPlus simulation environment. (Version 9.2 is required)

## Installation

Ensure you have Python and Jupyter Notebook installed. Using Anaconda is recommended for Windows.

Additional Python dependencies can be installed using:

```bash
pip install tensorflow numpy ...
```

## Usage

### Step-by-Step Execution:
1. **Run the Jupyter Notebook first** (`Main_Python.ipynb`) to initialize and train the policy gradient reinforcement learning agents, which will await MATLAB to connect via TCP/IP.

2. **Then run MATLAB scripts** (`Main_Matlab.m`) to start the co-simulation and connect to the Python scripts through the TCP/IP port.

**Important:** The Jupyter Notebook must run first, as it sets up and waits for TCP/IP connections initiated by MATLAB.

## Notes
- The data used to train the pretrained models located in folders `NN_PaperPretrain_i_V101_TE3` and `NN_PaperPretrain_v_V101_TE3` are not included. If you would like to train your own pretrained model, please contact me directly (bspark1102@gmail.com).

## Citation
If you use this code or methodology in your research, please cite our paper:

```
@article{PARK2023121364,
title = {Performance, robustness, and portability of imitation-assisted reinforcement learning policies for shading and natural ventilation control},
journal = {Applied Energy},
volume = {347},
pages = {121364},
year = {2023},
issn = {0306-2619},
doi = {https://doi.org/10.1016/j.apenergy.2023.121364},
author = {Bumsoo Park and Alexandra R. Rempel and Sandipan Mishra},
}
```

## Contact
For any questions or suggestions, please open an issue in this repository or directly contact me (bspark1102@gmail.com).
