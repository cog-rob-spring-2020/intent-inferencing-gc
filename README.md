# 16.412 Intent Inferencing Grand Challenge

This repository contains the code for the Intent-Inferencing Grand Challenge for 16.412 Spring 2020.

## Installation
```
git clone git@github.com:cog-rob-spring-2020/intent-inferencing-gc.git

cd intent-inferencing-gc

# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate

# Required packages
pip install numpy torch matplotlib PyQt5
pip install gif imageio
pip install pyparsing tqdm
```

## Data parsing
From `intent-inferencing-gc/src` run
```
python parse_data.py --data path_to_data --outpath path_to_outdir
```

To run the example dataset:
```
python parse_data.py --data data/carla_short_raw --outpath datasets/CARLA_short/
```

## Predicting Trajectories with CVM
```
cd constant_velocity_model
```
To get the ADE and FDE for all of the CARLA datasets:
```
python evaluate.py
```
To run with the angular velocity option
```
python evaluate.py --use_angvel
```

#### Generating Images
This script can only generate images for one dataset at a time. Edit `dataset_paths` in the `RunConfig` class at the top of `evaluate.py` to be a length 1 list.

Below are the different image generation options.

For one frame:
```
python evaluate.py --make_plot timestamp
```

For a gif:
```
python evaluate.py --save_gif path_to_gif
```

For a series of stills for each frame in the dataset:
```
python evaluate.py --save_imgs path_to_imgdir
```
