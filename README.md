# Deep Evidential Emotion Regression (DEER)

Code for "Estimating the Uncertainty in Emotion Attributes using Deep Evidential Regression". A general approach for tasks with subjective annotations.

## Setup
PyTorch == 1.11   
SpeechBrain == 0.5.13   

## Data prepartion
1. `data_preparation/msp-partition.py` -- prepare train/validation/test splits
2. `data_preparation/msp-label.py` -- prepare labels
3. `data_preparation/msp-data-json.py` -- prepare training scps  
    Example json file in msp-data/sample.json

## Training
`python3 DEER_train.py DEER_config.yaml --output_folder='exp'`  
  - Training log saved in exp/train_log.txt  
  - Model saved in exp/save  
  - Test predictions saved in exp/test_outcome-E{PLACEHOLDER}.npy  
  
  
  `DEER_train.py` -- training script  
  `DEER_config.yaml` -- training configuration  
  `deep_evidential_emotion_regression.py` -- DEER loss and evidential layer  
  `model.py` -- model class  
  `utils.py` -- metrics, sampler, etc.  

  \* Users are encouraged to experiment with different optimizers, schedulers, models, etc.

## Scoring
`python3 scoring.py exp/test_outcome-E{PLACEHOLDER}.npy`
