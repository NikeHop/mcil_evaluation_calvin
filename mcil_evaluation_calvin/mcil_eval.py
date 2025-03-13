""" Evaluate MCIL policy in CALVIN environment. """

import argparse

import torch
import wandb
import yaml

from calvin_agent.evaluation.evaluate_policy import evaluate_calvin, set_up_eval_config

from trainer import MCIL 

def evaluate_mcil_calvin(model, config):
    eval_config = set_up_eval_config(config, model)

    with torch.no_grad():
        evaluate_calvin(eval_config)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate MCIL policy in CALVIN environment.')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file.')
    args = parser.parse_args()

    # Load config 
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup wandb logging 
    wandb.init(
        project=config["logging"]["project"],
        mode=config["logging"]["mode"],
        tags=config["logging"]["tags"],
        name=config['logging']['experiment_name'],
        dir="../results",
    )
    # Load MCIL trainer 
    mcil = MCIL.load_from_checkpoint(args.checkpoint,device=config["device"]).to(config["device"])
    
    evaluate_mcil_calvin(mcil,config)