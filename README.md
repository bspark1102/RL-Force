# Imitation-assisted Reinforcement Learning for Dynamic Passive System Control

## Overview
This repository contains code and resources related to the paper "Performance, Robustness, and Portability of Imitation-assisted Reinforcement Learning for Dynamic Passive System Control". The goal of this research is to implement policy-gradient reinforcement learning to optimize the control of building ventilation and insulation systems dynamically.

## Structure
The repository consists of Python and MATLAB scripts, EnergyPlus input files, and utility functions necessary to replicate and extend experiments detailed in the associated research paper.

### Necessary Files

The following files are essential for running the simulations:

- **Policy Gradient Agents:**
  - `policygradient_tf3.py`: Base policy gradient agent.
  - `policygradient_tf3_dualI.py` & `policygradient_tf3_dualV.py`: Specialized dual agents for insulation and ventilation respectively.

- **Neural Network Models:**
  - `BP_NN_multi2.py`: Neural network definitions and training methods.

- **MATLAB Simulation Interface:**
  - `MLEP_PG_zzzPaper_train.m`: MATLAB-EnergyPlus co-simulation setup.
  - `MLEP_TCPIP_V6_MECC.m`: MATLAB script for managing TCP/IP communications with Python scripts.

- **Main Python Script:**
  - `PGpretrain_Paper_furthertrainTE3.ipynb`: Main Jupyter notebook for initializing and training reinforcement learning agents.

- **Utilities:**
  - `pushBack.m`, `setprod.m`: MATLAB helper functions for data manipulation.
  - `normalizer.m`, `normalizerV2.m`: Data normalization functions.

- **Reward and Action Functions:**
  - `new_reward_hvac4_action.m`: Custom reward calculation based on energy consumption.
  - `action_decouple.m`, `act2bin8.m`, `act2bin_decoupled_ins.m`, `act2bin_decoupled_ven.m`: Functions for translating action indices to actionable control signals.
  - `act7_select6_scratch_V1.m`, `act7_select6_sim.m`: Convert action indices into schedules for EnergyPlus.

- **EnergyPlus Models:**
  - `.idf` files define simulation models used by EnergyPlus (`031721_Single_NoSunspace...`, `070121j_Single_NoSunspace...`).

Files not listed above can be considered leftovers and are not essential for replicating the experiments.

## Requirements

- **Python Libraries:**
  - TensorFlow
  - NumPy
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
pip install tensorflow numpy
```

## Usage

### Step-by-Step Execution:
1. **Run the Jupyter Notebook first** (`PGpretrain_Paper_furthertrainTE3.ipynb`) to initialize and train the policy gradient reinforcement learning agents, which will await MATLAB to connect via TCP/IP.

2. **Then run MATLAB scripts** to start the co-simulation and connect to the Python scripts through the TCP/IP port:
   ```matlab
   run('MLEP_PG_zzzPaper_train.m')
   ```

**Important:** The Jupyter Notebook must run first, as it sets up and waits for TCP/IP connections initiated by MATLAB.

## Notes
- The data used to train the pretrained models located in folders `NN_PaperPretrain_i_V101_TE3` and `NN_PaperPretrain_v_V101_TE3` are not included.

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
For any questions or suggestions, please open an issue in this repository or contact the authors (bspark1102@gmail.com) directly.
