# Panoptes: A Multi-Resolution Convolutional Network for Cancer H&E Classification
This is a re-implementation of the original [Panoptes](https://github.com/rhong3/panoptes-he) in TensorFlow2.  

<br />

#### Step 0: Creating the environment
Clone the github repository and install the required python libraries/binaries into your local environment:
```
git clone https://github.com/Wenke-Liu/panoptes.git
conda create conda env create -f environment.yml
```

#### Step 1: Format your training, validation, and test splits. 
The training, validation, and test splits should be formatted prior to model training/evaluation. We do not use tfrecords in this implementation, and instead read tiles directly from png files. Each row represents a single paired instance of a 10x, 5x, and 2.5x tile. Each file should contain the following columns:

| Column Name  | Description |
| ------------- | ------------- |
| Patient_ID  | Any corresponding metadata identifying each patient.   |
| Slide_ID  | Any corresponding metadata identifying which slide within the patient.    |
| Tumor  | The tumor this tile pair originates from - weighted loss will be calculated within each tumor type.    |
| label  | A numerical label indicating the outcome (ex. 0 for normal and 1 for tumor, or 0,1,2,3,4 for clinical grade).  |
| L1path  | The file path to the 10x tile.    |
| L2path  | The file path to the 5x tile.   |
| L3path  | The file path to the 2.5x tile.    |





