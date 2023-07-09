# Deep Evidential Emotion Regression (DEER) :deer:

Code for "Estimating the Uncertainty in Emotion Attributes using Deep Evidential Regression".  


[Paper](https://aclanthology.org/2023.acl-long.873)  

Please cite:  

>@inproceedings{wu-etal-2023-estimating,  
>    title = "Estimating the Uncertainty in Emotion Attributes using Deep Evidential Regression",  
>    author = "Wu, Wen  and Zhang, Chao  and  Woodland, Philip",  
>    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",  
>    month = jul,  
>    year = "2023",  
>    address = "Toronto, Canada",  
>    publisher = "Association for Computational Linguistics",  
>    url = "https://aclanthology.org/2023.acl-long.873",  
>    pages = "15681--15695",  
>    }

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
