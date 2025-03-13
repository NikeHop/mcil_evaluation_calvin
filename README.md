# Evaluation MCIL policy in CALVIN

Code to evaluate MCIL policies trained for the CALVIN environment using the following [repo](https://github.com/NikeHop/PlaySegmentation-AAAI2025).

## Dependencies & Setup 

- Anaconda3/Miniconda3

Run the following script (~3hr) to:
- Creates conda environment `mcil_evaluation_calvin` with dependencies.
- Downloads the CALVIN dataset.

```
bash setup.sh
```

## Evaluate MCIL policy

From the `./mcil_evaluation_calvin` directory run the following command:

```
python train.py --config ./config/mcil_evaluation.yaml --checkpoint PATH_TO_CHECKPOINT 
```
