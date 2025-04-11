# PRML 2025 Project 4: Enhancement and Validation on Estimating Foot Pressure from Visual Input

## Accessing PSUTMM-100 Dataset (Important)

To get access to the PSUTMM-100 dataset, you must fill out the following form: [PSUTMM-100 Dataset Request Form](https://forms.office.com/pages/responsepage.aspx?id=RY30fNs9iUOpwcEVUm61LmRL31cpojBLmxyKYF0XS1tURVMxOFoxWFpaSzdDRFZZNkhXVlA4TEkyWS4u&route=shorturl)


## Project Description

Please first refer to the project description on Canvas for exact information. You may select from one of the following options:

1. **Option 1: Optimize the Existing Model**
   
    Start with the provided FootFormer baseline and focus on improving its performance through training-related strategies.

2. **Option 2: Propose/Develop a New Model**
   
    Design and implement a new architecture for the foot pressure estimation task. You can completely reimagine the encoder, temporal model, or output head.

3. **Option 3: Conduct Exploratory or Ablation Experiments**

    Explore how different parts of the model or input data affect performance. This path is well-suited if you’re interested in understanding model behavior through systematic analysis.

There are two extension directions for this project:

1. **Extension 1: Beyond Foot Pressure Prediction**

    In addition to foot pressure prediction, we extend the model to predict foot contact and Center of Mass. You may also propose other tasks for the model to predict and evaluate it on those datasets (GroundLink, UnderPressure, etc.)

2. **Extension 2: Estimating Stability**

    Using the predicted foot pressure and CoM predictions, we can quantify stability metrics and evaluate them to the ground truth values.

## Codebase Overview

The codebase is structured as follows:

- `scripts/`: Contains scripts for training, evaluation, and and dataset creation. See the relevant sections for more details.
- `pressure/`: Contains the implementation of the FootFormer model and other related components including
  - `data/`: Contains the dataset class and data processing utilities.
  - `models/`: Contains the FootFormer model and other related components.
  - `utils/`: Contains utility functions for training, evaluation, and visualization.

## Environment Setup

To set up an environment to run the code, we suggest using [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). You can create a new environment with

```bash
conda create -n footpressure python=3.10 -y
conda activate footpressure
```

You will then need to install [PyTorch](https://pytorch.org/get-started/locally/). You should visit the PyTorch website and select the appropriate installation command for your system. For example, to install PyTorch with CUDA 11.1, you can run the following command:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

Finally, you can install the other dependencies with and setup the codebase with:

```bash
pip install -e .
```

## Dataset Creation

After filling out the form to access the PSUTMM-100 dataset, you will receive a link to download the dataset. We will then use the `scripts/create_chunk_dataset.py` script to create the dataset in the required format. It will be up to you to look at the various arguments available to you if you wish to change any of the preprocessing steps. By default, to create a dataset using 3D OpenPose joints with corresponding foot pressure distributions, and Center of Mass values, you may run the following command:

```bash
python scripts/create_chunk_dataset.py --gt_com --make_pressure_distribution --save_path Chunked_PSU/BODY25_3D_5fps_distribution --root_dir path_to_dataset/PSU100/Subject_wise --center_w_op
```

where the above arguments
- `--gt_com` will include the ground truth Center of Mass in the dataset.
- `--make_pressure_distribution` will normalize the pressure maps into distributions.
- `--save_path` specifies the path to save the dataset.
- `--root_dir` specifies the root directory of the PSUTMM-100 dataset.
- `--center_w_op` will center the CoM about the OpenPose joints (and not the 3D mocap joints).

If you are not doing one of the extensions, you will only be using the foot pressure distribution data.

You  can find the Ordinary Movement data here: [OneDrive Link](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/kbk5531_psu_edu/EWnEc94Sp5NIj0XaGIf9cBkBYGadtD9smVOj05j1gD9L4g?e=OebffI)

## Training and Evaluation

To train the FootFormer model to predict pressure from the dataset, you can use the `scripts/train.py` script and pass a configuration file. You can run the following command to train the model:

```bash
python scripts/train.py --config configs/footformer.yaml
```

This will train and test the model LOSO (Leave-One-Subject-Out) on the PSUTMM-100 dataset and save the output, checkpoints, and evaluation results in the `output/` directory (specified in the configuration file). You can resume training with

```bash
python scripts/train.py --config configs/footformer.yaml --resume
```

You can also just evaluate a trained model with

```bash
python scripts/eval.py --config configs/footformer.yaml
```

## Evaluating on Ordinary Movements 

To evaluate on the ordinary movements dataset, you first will need to create the dataset with 

```bash
python scripts/create_chunk_dataset.py --make_pressure_distribution --save_path Chunked_PSU/BODY25_3D_5fps_distribution --root_dir path/to/OM/root --sample_rate 1
```

In a config file, set the `checkpoint` directory to the path of a checkpoint for **Subject1**. Then, you can run the evaluation script with

```bash
python scripts/eval.py --config configs/footformer_pressure_OM.yaml
```

## Visualization

To visualize test results, set `viz.enabled` to `True` in the config file. You can then run the evaluation on a subject/set you want to visualize. Visualization can be expensive. It will be up to you to look at what each of the parameters in the config file does and how to change them.
