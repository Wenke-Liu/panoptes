# Panoptes: A Multi-Resolution Convolutional Network for Cancer H&E Classification
This is a re-implementation of the original [Panoptes](https://github.com/rhong3/panoptes-he) in TensorFlow2.  

<br />

#### Step 0: Creating the environment.
Clone the github repository and install the required python libraries/binaries into your local environment:
```
git clone https://github.com/Wenke-Liu/panoptes.git
conda create conda env create -f environment.yml
```

#### Step 1: Format your training, validation, and test splits. 
The training, validation, and test splits should be formatted prior to model training/evaluation. We do not use tfrecords in this implementation, and instead read tiles directly from png files. All rows should be shuffled prior to the next step. Each row represents a single paired instance of a 10x, 5x, and 2.5x tile. Each file should contain the following columns:

| Column Name  | Description |
| ------------- | ------------- |
| Patient_ID  | Any corresponding metadata identifying each patient.   |
| Slide_ID  | Any corresponding metadata identifying which slide within the patient.    |
| Tumor  | The tumor this tile pair originates from - weighted loss will be calculated within each tumor type.    |
| label  | A numerical label indicating the outcome (ex. 0 for normal and 1 for tumor, or 0,1,2,3,4 for clinical grade).  |
| L1path  | The file path to the 10x tile.    |
| L2path  | The file path to the 5x tile.   |
| L3path  | The file path to the 2.5x tile.    |

<br />

#### Step 2: Training and evaluation. 
Models are trained from scratch and performance evaluated on the test set previously defined in Step 1. A description of the accepted parameters are as follows:
- ```--OUT_DIR```   
  Path to the directory for all output and intermediary log files.   
- ```--trn_df```   
  Path to the previously created table of training tile pairs.   
- ```--val_df```   
  Path to the previously created table of validation tile pairs.     
- ```--tst_df```   
  Path to the previously created table of testing tile pairs.    
- ```--contrastive```   
  Add this flag to pretrain the model with contrastive loss prior to training the classifier head.  
- ```--VAL_SAMPLE```   
  The number of validation samples that will be selected for intermediary model evaluation to determine early stopping. 
- ```--MANIFOLD_SAMPLE```   
  The number of test tile pairs that will be sampled for TSNE visualization.  
- ```--BATCH_SIZE```   
  The number of test tile pairs in each batch of training data. 
- ```--STEPS```   
  The number of steps to train prior to model evaluation on the validation subset. Each step is the size of one batch. 
- ```--MAX_EPOCH```   
  The maximun number of epochs to train for. Each epoch is the size of the number of steps previously defined.  
- ```--PATIENCE```   
  The number of epochs to wait for validation loss to decrease before early stop of training.   
- ```--SEED```   
  The seed used to determine the subsetting of validation and test samples - added to improve replicability.   
- ```--MULTI_GPU```   
  Include this flag for training across multi-GPU devices.  

For training, Panoptes supports adding a contrastive pre-training step prior to training the classifier:  
  - If the ```--contrastive``` flag is passed to main.py, then a nonlinear 128-dim projection head is attached to the top of the encoder, and first pre-trained to optimize the contrastive loss. Thereafter, the projection head is removed and replaced with a Dense prediction layer with size corresponding to the number of class outcomes. The weights of the previously pre-trained encoder are not frozen during training of the prediction layer.
  - By default, not passing the ```--contrastive``` will only train a classification with no contrastive pre-training.    

Panoptes will automatically create the following files within each directory at ```--OUT_DIR```:

- data   
  \*_slide_idx.csv and \*_tile_idx.csv where \* indicates trn, val, and tst for the trainig, validation, and test splits created earlier.   
- log  
  If ```--contrastive``` is enabled, then pre_trn_history_logs.csv will contain the loss values, while tensorboard-compatible viewing is available within pre_trn_tb_logs. Similarly, trn_history_logs.csv and trn_tb_logs contain information for the classification training.  
- model  
  Final model weights are saved as "panoptes_weights_final.h5" and intermediary checkpoints within ckpt directory.  
- pred  
  Predictions for each individual slide and aggregated at the slide level are saved as tst_tile_pred.csv and tst_slide_pred.csv respectively. Coordinates of the tsne clustering are saved in tSNE_P_N.csv.
  

#### Step 3: Reloading a trained model for external test. 






