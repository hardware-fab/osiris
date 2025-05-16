# Osiris: A Scalable Dataset Generation Pipeline for Machine Learning in Analog Circuit Design

**Osiris** is an end-to-end analog circuit design pipeline capable of producing, validating, and evaluating large volumes of layouts for generic analog circuits.

The [Osiris GitHub repository](https://github.com/hardware-fab/osiris) hosts the code that implements the randomized pipeline as well as the reinforcement learning-driven baseline methodology discussed 
in the paper proposed at the NeurIPS 2025 Datasets & Benchmarks Track.

The [Osiris 🤗 HuggingFace repository](https://huggingface.co/datasets/hardware-fab/osiris) hosts the randomly generated dataset discussed in the paper.

In this repository, there is the Osiris Python script (`osiris.py`) and the Python friendly version of the SkyWater 130nm PDK (`SKY130_PDK`) along with the necessary files to perform place and route operations (in `schematic2layout` folder).

## Installation
To run Osiris a suite of tools is necessary. 
### ALIGN
To install ALIGN clone its repository:
```bash
git clone git@github.com:ALIGN-analoglayout/ALIGN-public.git
```
and apply the `align.patch` located in this repository:
```bash
git apply align.patch
```
Finally install it:
```bash
pip install -v .
```

### Netgen
Netgen is used to carry out Layout Versus Schematic (LVS). We use version `1.5.293`.
To install it, please refer to the official GitHub [repository](https://github.com/RTimothyEdwards/netgen).

### Magic
Magic is used to perform Parasitic Extraction (PEX). We use version `8.3.497`.
To install it, please refer to the official GitHub [repository](https://github.com/RTimothyEdwards/magic).

### Ngspice
Ngspice is used to perform simulations. We use version `ngspice-43`.
To install it, please refer to the official [website](https://ngspice.sourceforge.io/).

### SkyWater 130nm PDK
To install the SkyWater 130 nm PDK used throughout the research, please refer to the official [GitHub repository](https://github.com/google/skywater-pdk).
These PDK files are used by Netgen, Magic, and Ngspice to perform LVS, PEX, and simulation respectively. While the `SKY130_PDK` files are 
a Python friendly version of the SkyWater 130nm PDK to perform the place and route operations.

## Usage
You will need to adjust a few paths to our working setup. 
- in each testbench: the path pointing to SkyWater simulation libraries, i.e., `sky130A/libs.tech/ngspice/sky130.lib.spice`.
- in `osiris.py`: the `WD` global variable.
- in `0_alter_nf.py`: the `base_path` local variable.
- in `pex.sh`: the path pointing to SkyWater root, i.e., `sky130A`.
- in `1_reroute_sch2lay.sh`: the path pointing to `SKY130_PDK`.

Then you will need to compute the fingers permutations. 
To do so, refer to the `2_flow/0_alter/0_alter_nf.py` script.

To experiment with Osiris, use the `osiris.py` file.

To perform random exploration of circuit design space, set: 

> EXPLORATION_MODE = 'dataset'

To perform RL-driven exploration of circuit design space, set:

> EXPLORATION_MODE = 'rl'

## Note
This repository is protected by copyright and licensed under the [Apache-2.0 license](https://github.com/hardware-fab/chameleon/blob/main/LICENSE) file.

© 2025 hardware-fab
