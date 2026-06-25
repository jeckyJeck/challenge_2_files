# Intro to ML Challenge 2

This repository contains our team's submission for Challenge 2.

## How to train the model

We have updated the `train.py` script to automatically look for a `config.json` configuration file in its directory. This allows you to easily run the code in different environments (like Google Colab) by pointing to your specific data and model paths.

### 1. Configure paths
The script will look for `config.json` in the `submissions/my_team` directory. If it exists, it will use the paths and hyperparameter overrides. If you omit any parameter, it will fall back to the default value.

```json
{
    "train_data_path": "../../dataset/train_set",
    "output_weights_path": "weights.joblib",
    "model_architecture_path": "model.py",
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001
}
```

### 2. Run the Script
To run the training script from the root directory using the batch file:

```bash
run_train.bat
```

Or run it directly from the `my_team` directory:

```bash
cd submissions/my_team
python train.py
```

### Notes
- `train_data_path`: Overrides the default dataset location.
- `output_weights_path`: Where the trained weights (`weights.joblib`) will be saved.
- `model_architecture_path`: If provided, dynamically imports your `ModelArchitecture` from this specific python file.
