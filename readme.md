preparation

# Learning Editable/Adaptive High-Fidelity Animatable avatar from Casual videos

## Description

![img](assets/teaser.png)

Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.Abstract.

## Usage

### Installation

Please run the following scripts for create python virtual environment and install the dependencies.

```
conda env create -f environment.yml

# build and install dependencies
bash scripts/install.sh
```

### SMPL Setup

Download `SMPL v1.0 for Python 2.7` from [SMPL website](https://smpl.is.tue.mpg.de/) (for male and female models), and `SMPLIFY_CODE_V2.ZIP` from [SMPLify website](https://smplify.is.tue.mpg.de/) (for the neutral model). After downloading, inside `SMPL_python_v.1.0.0.zip`, male and female models are `smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl` and `smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl`, respectively. Inside `mpips_smplify_public_v2.zip`, the neutral model is `smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`. Remove the chumpy objects in these .pkl models using [this code](https://github.com/vchoutas/smplx/tree/master/tools) under a Python 2 environment (you can create such an environment with conda). Finally, rename the newly generated .pkl files and copy them to subdirectories under `./data/body_models/smpl/`. Eventually, the `./data/body_models` folder should have the following structure:

```
data/body_models
 └-- smpl
    ├-- male
    |   └-- model.pkl
    ├-- female
    |   └-- model.pkl
    └-- neutral
        └-- model.pkl

```

Then, run the following script to extract necessary SMPL parameters used in our code:

```
python extract_smpl_parameters.py
```

The extracted SMPL parameters will be saved into `./body_models/misc/`.

### Data Preparation

You can get the raw data from their respective sources and use our preprocessing script to generate data that is suitable for our training/validation scripts. Please follow the steps in [DATASET.md](https://github.com/taconite/arah-release/blob/main/DATASET.md). Train/val splits on cameras/poses follow [NeuralBody&#39;s split](https://github.com/zju3dv/neuralbody/blob/master/supplementary_material.md#training-and-test-data). Pseudo ground truths for geometry reconstruction on the ZJU-MoCap dataset are stored in [this folder](https://drive.google.com/drive/folders/1-OE3h7nxg7ixL3yh0Y7bGYKVsNWS-Zm4?usp=share_link). For evaluation script and data split of geometry reconstruction please refer to [this comment](https://github.com/taconite/arah-release/issues/9#issuecomment-1359209138).

### Training

```
bash scripts/run_377_neus.sh
```

### Evaluation

```
bash scripts/run_377_neus.sh
```

## Log

1. 代码可以跑通了，但是结果还不好，需要调

## Citation

```
@article{wang2023b,
  title={TITLE},
  author={AUTHOR},
  journal={JOURNAL},
  year={YEAR}
}
```

## License

Distributed under the GPL License. See `LICENSE` for more information.

## Acknowledgements

This project is built on [ARAH](https://github.com/taconite/arah-release), thank them for their contributions.
