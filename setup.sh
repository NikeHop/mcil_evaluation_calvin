#! /bin/bash

set -e 

# Download the dataset
mkdir -p ./datasets/calvin

cd ./datasets/calvin


if [ -f "./calvin_debug_dataset.zip" ]; then
    echo "File already exists, no download needed"
else
    echo "Downloading debug dataset ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
fi

if [ -d "./calvin_debug_dataset" ]; then
    echo "Already unzipped, no need to unzip again"
else
    unzip calvin_debug_dataset.zip
    echo "saved folder: calvin_debug_dataset"
    rm calvin_debug_dataset.zip
fi


if [ -f "./task_D_D.zip" ]; then
    echo "File already exists, no download needed"
else
    echo "Downloading debug dataset ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
fi

if [ -d "./task_D_D" ]; then
    echo "Already unzipped, no need to unzip again"
else
    unzip task_D_D.zip
    echo "saved folder: task_D_D"
    rm task_D_D.zip
fi


# Back to base dir 
cd ../../

# Create conda environment 
source $CONDA_PREFIX/bin/activate
conda create -n mcil_evaluation_calvin_2 python=3.8
conda activate mcil_evaluation_calvin_2


cd ./mcil_evaluation_calvin/calvin/calvin_env/tacto
pip install -e . 
cd ..
pip install -e .
cd ..
cd  ./calvin_models
pip install -e .
cd ../../..
pip install -e .