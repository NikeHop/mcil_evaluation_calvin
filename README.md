# Evaluation MCIL policy in CALVIN
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Code to evaluate MCIL policies trained for the CALVIN environment using the following [repo](https://github.com/NikeHop/PlaySegmentation-AAAI2025).

## Dependencies & Setup 

Clone the repo with its submodule

```
git clone --recursive https://github.com/NikeHop/mcil_evaluation_calvin.git 
```

- Anaconda3/Miniconda3

Run the following bash script (~3hr) to:
- Downloads the CALVIN dataset.
- Creates conda environment `mcil_evaluation_calvin` with dependencies.

```
yes | bash setup.sh
```

## Evaluate MCIL policy

From the `./mcil_evaluation_calvin` directory run the following command:

```
python train.py --config ./config/mcil_evaluation.yaml --checkpoint PATH_TO_CHECKPOINT 
```
