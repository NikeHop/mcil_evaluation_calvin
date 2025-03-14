# Evaluation MCIL policy in CALVIN

Code to evaluate MCIL policies trained for the CALVIN environment using the following [repo](https://github.com/NikeHop/PlaySegmentation-AAAI2025).

## Dependencies & Setup 

Clone the repo with its submodule

```
git clone --recursive https://github.com/NikeHop/mcil_evaluation_calvin.git 
```

- Anaconda3/Miniconda3

Run the following commands (~3hr) to:
- Downloads the CALVIN dataset.
- Creates conda environment `mcil_evaluation_calvin` with dependencies.

```
bash setup.sh
conda create -n mcil_evaluation_calvin_2 python=3.8
conda activate mcil_evaluation_calvin_2
pip install -e .
cd ./mcil_evaluation_calvin/calvin_env/tacto
pip install -e .
cd ../
pip install -e .
cd ../calvin_models
pip install -e . 
```

## Evaluate MCIL policy

From the `./mcil_evaluation_calvin` directory run the following command:

```
python train.py --config ./config/mcil_evaluation.yaml --checkpoint PATH_TO_CHECKPOINT 
```
