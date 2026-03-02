# CAP5516 - Assignment 1

This script trains and evaluates a ResNet18 architecture on the Chest-Xray Dataset for Pneuomonia classification. The dataset can be found on [this kaggle repository](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## Environment Setup
You can use other virtual environmenet managers. This codebase, however, was tested using anaconda.

Run the following commands to create the conda environment:
```
conda create -n cap5516_ass1 python=3.11
conda activate cap5516_ass1
pip install -r requirements.txt
```

## Training
To train the model, run the train script as follows:
```
python train.py \
    --root <dataset_root_dir> \
    --log_dir <directory where logs will be saved, default is "logs/"> \
    --batch_size <default=32> \
    --lr <default=0.0001> \
    --epochs <default=10> \
    --use_pretraiend # use this if you want to use imagenette pre-trained weights for ResNet18
```

The best model would be saved as `best.pt` inside the provides logs directory. The script would also save the graphs in the provided logs directory for loss (train and validation), accuracy (train and validation), and F1-score.
A `training_logs.json` file is also stores in hte log directory. It contains information about the validation and training metrics over epochs, as well as the test performance (loss, f1-score, accuracy) evaluate using the best performing model (determined by validation loss during training).

## Testing
The test script takes a trained model and evaluates it on the provided test set. It also uses GradCAM to generate heatmaps for the images, saving them in the the cams/ folder inside the provided logs directory.
To test the trained model, use the following command:
```
python test.py \
    --root <dataset_root_dir> \
    --model_path <path to the trained model> \
    --log_dir <directory where CAMs will be saved "logs/"> \
    --batch_size <default=32> \
```
