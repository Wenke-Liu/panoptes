# Panoptes: A Multi-Resolution Convolutional Network for Cancer H&E Classification
This is a re-implementation of the original [Panoptes](https://github.com/rhong3/panoptes-he) in TensorFlow2.  

<br />

### Step 0: Creating the environment.
Clone the github repository and install the required python libraries/binaries into your local environment:
```
git clone https://github.com/Wenke-Liu/panoptes.git
conda create conda env create -f environment.yml
```

### Step 1: Format your training, validation, and test splits. 
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

### Step 2: Training and Validation. 
Models are trained from scratch and performance evaluated on the test set previously defined in Step 1 using ```train.py```.  
A description of the accepted parameters are as follows:
```
Usage: python train.py [OPTIONS]
  --OUT_DIR         Path to the directory for all output and intermediary log files. Do not append "/" to this path.  
  --trn_df          Path to the previously created table of training tile pairs.   
  --val_df          Path to the previously created table of validation tile pairs.     
  --tst_df          Path to the previously created table of testing tile pairs.    
  --contrastive     Add this flag to pretrain the model with contrastive loss prior to training the classifier head.  
  --num_classes     Specify the number of outcome classes for prediction - will only affected the Dense prediction layer in the classifier head.  
  --VAL_SAMPLE      The number of validation samples that will be selected for intermediary model evaluation to determine early stopping. 
  --MANIFOLD_SAMPLE The number of test tile pairs that will be sampled for TSNE visualization.  
  --BATCH_SIZE      The number of test tile pairs in each batch of training data. 
  --STEPS           The number of steps to train prior to model evaluation on the validation subset. Each step is the size of one batch. 
  --MAX_EPOCH       The maximun number of epochs to train for. Each epoch is the size of the number of steps previously defined.  
  --PATIENCE        The number of epochs to wait for validation loss to decrease before early stop of training.   
  --SEED            The seed used to determine the subsetting of validation and test samples - added to improve replicability.   
  --MULTI_GPU       Include this flag for training across multi-GPU devices.  
```

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

### Examples:  
Sample code to run the following objectives:  
-  Model training with contrastive pretraining, followed by classifier training  
-  Using the example files in the example folder  
-  Classify normal vs. tumor.  
-  Batch size of 24 
-  Use 20,000 tile pairs selected for tsne evaluation.  

```python train.py --contrastive --trn_df examples/trn_df.csv --val_df examples/val_df.csv --tst_df examples/tst_df.csv --num_classes 2 --MANIFOLD_SAMPLE 20000 --BATCH_SIZE 24``` 

In contrast, to run the exact same settings above without contrastive pretraining, simply remove the ```--contrastive``` flag:  

```python train.py --trn_df examples/trn_df.csv --val_df examples/val_df.csv --tst_df examples/tst_df.csv --num_classes 2 --MANIFOLD_SAMPLE 20000 --BATCH_SIZE 24``` 

<br />

### Step 3: Reloading a trained model for external test.  
Models can be reloaded back into the environment for testing on a different dataset using ```test.py```.  
A description of the accepted parameters are as follows:  
```
Usage: python test.py [OPTIONS]. 
  --OUT_DIR         Path to the directory for all output and intermediary log files. Do not append "/" to this path.   
  --MODEL_WEIGHTS   File path to the previously created .h5 weights file, or any other .h5 weights file created in Step 2.   
  --num_classes     The number of outcome classes that was originally used to train the above model.  
  --tst_df          Path for the external test dataset.  
```      

The following pre-trained models used in our publication can be downloaded and loaded into Step 3:  
| Model Outcome  | num_classes | link |
| ------------- | ------------- | ------------- |
| Normal vs. Tumor  | 2 (Normal, Tumor)  |  [Google Drive](https://drive.google.com/file/d/19ovu1oMGvscNpo-PGaYQ6DcAA_J8hXVe/view?usp=share_link) |  
| Tissue-of-Origin  | 6 (CCRCC, HNSCC, LSCC, LUAD, PDA, UCEC)  |  [Google Drive](https://drive.google.com/file/d/19hemZ8OukAsKBZrLelzzp-c8AIbJ_Uc3/view?usp=share_link) |  
| Clinical Grade  | 5 (Normal, 1, 2, 3, 4)  |  [Google Drive](https://drive.google.com/file/d/19Zp98zQBK7P5NdVUBg6R38M2_CfgJ3oV/view?usp=share_link) |  
| Clinical Stage  | 5 (Normal, 1, 2, 3, 4)  |  [Google Drive](https://drive.google.com/file/d/19iuT0Q3Y9DHjGmgR-E2K3gLPdqy1tkP6/view?usp=share_link) |  
| Percent Tumor Nuclei  | 3 (Low, Medium, High)  |  [Google Drive](https://drive.google.com/file/d/19h_A_8f9BWUJhOCdOhJtffawtoTN9SwR/view?usp=share_link) |  
| Percent Tumor Cellularity  | 3 (Low, Medium, High)  |  [Google Drive](https://drive.google.com/file/d/19ah6omW-8dTLXE9d49FmnH3fh4UDZXf6/view?usp=share_link) |  
| Percent Necrosis  | 3 (Low, Medium, High)  |  [Google Drive](https://drive.google.com/file/d/19gT9an4faSPAPOiXvIchLwkkFYH_jeyQ/view?usp=share_link) |  
| TP53  | 2 (Wild-type, Mutated)   |  [Google Drive](https://drive.google.com/file/d/19VAZFkQLGJqhOTuzZTIjrXS5B7vBvX-8/view?usp=share_link) |  
| EGFR  |  2 (Wild-type, Mutated)   |  [Google Drive](https://drive.google.com/file/d/19Pn0pFAQ22LBd-w0h5oFiPo7ARa7LB8k/view?usp=share_link) |  
| KRAS  |  2 (Wild-type, Mutated)   |  [Google Drive](x) |  
| PTEN  |  2 (Wild-type, Mutated)   |  [Google Drive](https://drive.google.com/file/d/19SD8umCXbFxxehHR7jXTwbbn1k2Nurmt/view?usp=share_link) |  
| STK11  |  2 (Wild-type, Mutated)   |  [Google Drive](https://drive.google.com/file/d/19XJS32XQLzCbZow7jn7Pyhl21OpM17QS/view?usp=share_link) |    

The following files will be created at ```--OUT_DIR```:  
- test_preds_tile.csv: prediction values for each tile in tst_df.  
- test_latent_tile.csv: latent vectors for each tile in tst_df.  
- test_preds_image.csv: prediction values averaged for each slide's tile.  

### Examples:   
To import your model and test on the example tst_df.csv:  
```python test.py --MODEL_WEIGHTS your_model_weights.h5 --num_classes 2 --tst_df examples/tst_df.csv```   

Alternatively, to use our model (ex. normal vs. tumor) on the example tst_df.csv:        
```
Download the above Normal vs. Tumor model from the Google Drive link, then:  
python test.py --MODEL_WEIGHTS tumor_CCA_weights_converted.h5 --num_classes 2 --tst_df examples/tst_df.csv
```
   





