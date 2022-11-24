from data_input import *
from model import *
from utils import *
import pandas as pd
import numpy as np 

import argparse
parser = argparse.ArgumentParser()
#Universal arguments
parser.add_argument('--OUT_DIR',type=str,default='./results/tp53_test_221018') 
parser.add_argument('--MODEL_WEIGHTS',type=str,default='./model/panoptes_weights_final.h5')
parser.add_argument('--num_classes',type=int,default=2)
parser.add_argument('--tst_df',type=str,default='te_sample_TCGA.csv')

args = parser.parse_args()

tst_df = pd.read_csv(args.testInputFile)
tst_df['sample_weights'] = 1  # unweighted

tst = DataSet(filenames=tst_df[['L1path', 'L2path', 'L3path']],
              labels=tst_df['label'], 
              tile_weights=tst_df['sample_weights'], id_level=3)

tst_ds = tst.create_dataset(shuffle=False, batch_size=32, ds_epoch=1)

model = PANOPTES(contrastive=False,saved_model=args.modelWeights,n_classes=args.num_classes)
tst_res = model.inference(tst_ds)

latent = tst_res[0]
pred = tst_res[1]

pred = pd.DataFrame(pred)
predImage = pred.copy()

pred['Slide_ID'] = tst_df['Slide_ID']
pred['Patient_ID'] = tst_df['Patient_ID']
pred['Tumor'] = tst_df['Tumor']
pred['label'] = tst_df['label']
pred.to_csv(args.OUT_DIR+'/test_preds_tile.csv',index=False)

latent = pd.DataFrame(latent)
latent['Patient_ID'] = tst_df['Patient_ID']
latent['Slide_ID'] = tst_df['Slide_ID']
latent['Tumor'] = tst_df['Tumor']
latent['label'] = tst_df['label']
latent.to_csv(args.OUT_DIR+'/test_latent_tile.csv',index=False)


predImage['Patient_Slide_ID'] = pred['Patient_ID']+'-'+pred['Slide_ID']
predImage = predImage.groupby(['Patient_Slide_ID']).mean()

lookupDict = dict(zip(tst_df['Patient_ID']+'-'+tst_df['Slide_ID'],tst_df['label']))
test_labels_image = [lookupDict[i] for i in predImage.index]

predImage['Patient_Slide_ID'] = predImage.index
predImage['label'] = test_labels_image
predImage.to_csv(args.OUT_DIR+'/test_preds_image.csv',index=False)