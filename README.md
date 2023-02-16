# Deep Evidential Emotion Regression (DEER)

Code for "Estimating the Uncertainty in Emotion Attributes using Deep Evidential Regression", a general approach for subjective tasks where a sample can have multiple different annotations.

## Setup
PyTorch >= 1.11 (might also work on older versions)  
SpeechBrain >= 0.5.13 (recommend using the latest version on [github](https://github.com/speechbrain/speechbrain))  
numpy  
json  

## Data prepartion:
1. `python3 data_preparation/msp-partition.py` -- prepare train/validation/test splits
2. `python3 data_preparation/msp-label.py` -- prepare labels
3. `python3 data_preparation/msp-data-json.py` -- prepare training scps  
    Example json file in msp-data/sample.json

## Training
`python3 DEER_train.py DEER_config.yaml --output_folder='exp'`  
  - Training log saved in exp/train_log.txt  
  - Model saved in exp/save  
  - Test predictions saved in exp/test_outcome-E{PLACEHOLDER}.npy  
  
  
  `DEER_train.py` -- training script  
  `DEER_config.yaml` -- training configuration  
  `model.py` -- model class  
  `utils.py` -- metrics, sampler, etc.  

  \* Training 110+ hours of MSP-Podcast data took aroud 5 hours on 1 NVIDIA A100 GPU.  
  \* Users are encouraged to experiment with different optimizers, schedulers, models, etc.

Scoring
`python3 scoring.py exp/test_outcome-E{PLACEHOLDER}.npy`
