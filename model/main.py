import os
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from math import ceil
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())
from tensorflow.keras.utils import to_categorical
from data_input import DataSet
from model import PANOPTES
from sklearn.manifold import TSNE
from sklearn import metrics

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--OUT_DIR',type=str,default='./results/tp53_test_221018') 
parser.add_argument('--VAL_SAMPLE',type=int,default=10000)
parser.add_argument('--MANIFOLD_SAMPLE',type=int,default=20000)
parser.add_argument('--BATCH_SIZE',type=int,default=16)
parser.add_argument('--STEPS',type=int,default=10000)
parser.add_argument('--MAX_EPOCH',type=int,default=20)
parser.add_argument('--PATIENCE',type=int,default=2)
parser.add_argument('--SEED',type=int,default=221018)
parser.add_argument('--MULTI_GPU',action='store_true')

args = parser.parse_args()

OUT_DIR = args.OUT_DIR
VAL_SAMPLE = args.VAL_SAMPLE
MANIFOLD_SAMPLE = args.MANIFOLD_SAMPLE
BATCH_SIZE = args.BATCH_SIZE
STEPS = args.STEPS
MAX_EPOCH = args.MAX_EPOCH
PATIENCE = args.PATIENCE
SEED = args.SEED

MULTI_GPU = args.MULTI_GPU

print('Experiment random seed: ' + str(SEED))
np.random.seed(SEED)
SPLIT_SEED, VAL_SEED, TST_SEED = np.random.randint(low=0, high=1000000, size=3)
#print('Data split seed: ' + str(SPLIT_SEED))
print('Validation sample seed: ' + str(VAL_SEED))
print('Manifold sample seed: ' + str(TST_SEED))

try:
    os.makedirs(OUT_DIR + '/data')
    os.makedirs(OUT_DIR + '/pred')
except FileExistsError:
    pass


trn_df = pd.read_csv('idx_files/tp53/tr_sample.csv')
#trn_df = trn_df.sample(n=2000, random_state=VAL_SEED)    # sample
save_idx_df(out_dir=OUT_DIR + '/data', idx_df=trn_df, fn='trn')

print('Applying training sample weights:')
trn_df = stratified_weights(trn_df)

print('Number of training examples: ' + str(trn_df.shape[0]))
MAX_STEPS = ceil(trn_df.shape[0]/BATCH_SIZE)
print('Maximum steps per epoch: ' + str(MAX_STEPS))

val_df = pd.read_csv('idx_files/tp53/va_sample.csv')
val_df = val_df.sample(n=VAL_SAMPLE, random_state=VAL_SEED)    # sample
val_df['sample_weights'] = 1  # unweighted
save_idx_df(out_dir=OUT_DIR + '/data', idx_df=val_df, fn='val')

tst_df = pd.read_csv('idx_files/tp53/te_sample.csv')
#tst_df = tst_df.sample(n=VAL_SAMPLE, random_state=TST_SEED)    # sample
tst_df['sample_weights'] = 1  # unweighted
save_idx_df(out_dir=OUT_DIR + '/data', idx_df=tst_df, fn='tst')

trn = DataSet(filenames=trn_df[['L1path', 'L2path', 'L3path']],
              labels=trn_df['label'], 
              tile_weights=trn_df['sample_weights'], id_level=3)
val = DataSet(filenames=val_df[['L1path', 'L2path', 'L3path']],
              labels=val_df['label'], 
              tile_weights=val_df['sample_weights'],id_level=3)
tst = DataSet(filenames=tst_df[['L1path', 'L2path', 'L3path']],
              labels=tst_df['label'], 
              tile_weights=tst_df['sample_weights'], id_level=3)

trn_ds = trn.create_dataset(batch_size=BATCH_SIZE, ds_epoch=None)
val_ds = val.create_dataset(batch_size=BATCH_SIZE, ds_epoch=1)
tst_ds = tst.create_dataset(shuffle=False, batch_size=BATCH_SIZE, ds_epoch=1)


if MULTI_GPU:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = PANOPTES(contrastive=True, dropout=0.5)
        #model.compile(loss_fn=tf.keras.losses.CategoricalCrossentropy())

else:
    model = PANOPTES(contrastive=True, dropout=0.5)
    #model.compile(loss_fn=tf.keras.losses.CategoricalCrossentropy())

model.train(trn_data=trn_ds, val_data=val_ds,
            contrastive_temp=0.05,
            classifier_loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            steps=min(STEPS, MAX_STEPS),
            n_epoch=MAX_EPOCH,
            patience=PATIENCE,
            log_dir=OUT_DIR + '/log',
            model_dir=OUT_DIR + '/model')

model.print_attr()

#tst_res = model.inference(tst_ds)
print('Starting inference...')

tst_res = model.inference(tst_ds)

tst_df['NEG_Score'] = tst_res[1][:, 0]
tst_df['POS_Score'] = tst_res[1][:, 1]
tst_df.to_csv(OUT_DIR + '/pred/tst_tile_pred.csv', index=False)

tst_df_pred = tst_df[['Patient_ID', 'Slide_ID', 'label', 'Tumor', 'NEG_Score','POS_Score']]  # 'Tumor_Normal'
tst_df_pred_slide = tst_df_pred.groupby(['Patient_ID', 'Slide_ID', 'Tumor']).agg('mean').reset_index()  # 'Tumor_Normal'
tst_df_pred_slide.to_csv(OUT_DIR + '/pred/tst_slide_pred.csv', index=False)

fpr, tpr, thresholds = metrics.roc_curve(tst_df_pred_slide['label'], tst_df_pred_slide['POS_Score'], pos_label=1)
print('Slide level AUROC on test data: ' +
      str(metrics.auc(fpr, tpr)))


print('Generating tSNE...')
np.random.seed(TST_SEED)
sample_idx = np.random.choice(tst_df.shape[0], MANIFOLD_SAMPLE)    # sample 20,000 tiles for TSNE
print('Activation shape: ' + str(tst_res[0].shape))
tst_sampled = tst_res[0][sample_idx, :]
tst_df_sampled = tst_df.copy()
tst_df_sampled = tst_df_sampled.iloc[sample_idx, :]

tsne_embedding = TSNE(n_components=2).fit_transform(tst_sampled)

tst_df_sampled['tsne_0'] = tsne_embedding[:, 0]
tst_df_sampled['tsne_1'] = tsne_embedding[:, 1]
tst_df_sampled['NEG_Score'] = tst_res[1][sample_idx, 0]
tst_df_sampled['POS_Score'] = tst_res[1][sample_idx, 1]

tst_df_sampled.to_csv(OUT_DIR + '/pred/tSNE_P_N.csv', index=False)



