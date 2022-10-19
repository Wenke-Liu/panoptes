import os
import pandas as pd
import sklearn.utils as sku
import numpy as np
import re


def data_split(df,
               id='Patient_ID',
               stratify='Tumor',
               split_ratio=(0.8, 0.1, 0.1),
               collapse=True,
               seed=42):

    print('Using {} as id column.'.format(str(id)))
    split_points = np.cumsum(split_ratio)
    levels = df[stratify].unique()
    levels.sort()
    trn = []
    val = []
    tst = []
    trn_size = []
    val_size = []
    tst_size = []
    np.random.seed(seed)
    seeds = np.random.randint(low=0, high=1000000, size=len(levels))

    for i, level in enumerate(levels):    # stratified splits
        ids = df.loc[df[stratify] == level][id].unique()
        print('{} unique ids in {}'.format(str(len(ids)), str(level)))
        np.random.seed(seeds[i])
        prob = np.random.uniform(0, 1, size=len(ids))
        sub_df = pd.DataFrame({'ids': ids, 'prob': prob})
        trn.append(sub_df.loc[sub_df['prob'] < split_points[0]]['ids'])
        val.append(sub_df.loc[(sub_df['prob'] > split_points[0]) & (sub_df['prob'] < split_points[1])]['ids'])
        tst.append(sub_df.loc[sub_df['prob'] > split_points[1]]['ids'])
        trn_size.append(trn[i].size)
        val_size.append(val[i].size)
        tst_size.append(tst[i].size)

    print('Training samples: ' + str(trn_size))
    print('Validation samples: ' + str(val_size))
    print('Testing samples: ' + str(tst_size))

    if collapse:
        print('Collapsing ids in each split.')
        trn = pd.concat(trn)
        val = pd.concat(val)
        tst = pd.concat(tst)

    return trn, val, tst


def paired_tile_ids_in(patient, slide, tumor, root_dir):
    dira = os.path.isdir(root_dir + '/level1')
    dirb = os.path.isdir(root_dir + '/level2')
    dirc = os.path.isdir(root_dir + '/level3')
    if dira and dirb and dirc:
        fac = 1000
        ids = []
        for level in range(1, 4):
            dirr = root_dir + '/level{}'.format(str(level))
            for id in os.listdir(dirr):
                if '.png' in id:
                    x = int(float(id.split('x-', 1)[1].split('-', 1)[0]) / fac)
                    y = int(float(re.split('.p', id.split('y-', 1)[1])[0]) / fac)
                    ids.append([patient, slide, tumor, level, dirr + '/' + id, x, y])
        ids = pd.DataFrame(ids, columns=['Patient_ID', 'Slide_ID', 'Tumor', 'level', 'path', 'x', 'y'])
        idsa = ids.loc[ids['level'] == 1]
        idsa = idsa.drop(columns=['level'])
        idsa = idsa.rename(index=str, columns={"path": "L1path"})
        idsb = ids.loc[ids['level'] == 2]
        idsb = idsb.drop(columns=['Patient_ID', 'Slide_ID', 'Tumor', 'level'])
        idsb = idsb.rename(index=str, columns={"path": "L2path"})
        idsc = ids.loc[ids['level'] == 3]
        idsc = idsc.drop(columns=['Patient_ID', 'Slide_ID', 'Tumor', 'level'])
        idsc = idsc.rename(index=str, columns={"path": "L3path"})
        idsa = pd.merge(idsa, idsb, on=['x', 'y'], how='left', validate="many_to_many")
        idsa['x'] = idsa['x'] - (idsa['x'] % 2)
        idsa['y'] = idsa['y'] - (idsa['y'] % 2)
        idsa = pd.merge(idsa, idsc, on=['x', 'y'], how='left', validate="many_to_many")
        idsa = idsa.drop(columns=['x', 'y'])
        idsa = idsa.dropna()
        idsa = sku.shuffle(idsa)
    else:
        print('Pass: ', root_dir)  # exclude dir doesn't have all three levels
        idsa = pd.DataFrame(columns=['Patient_ID', 'Slide_ID', 'Tumor', 'L1path', 'L2path', 'L3path'])

    return idsa


def save_idx_df(out_dir, idx_df, fn):

    idx_df.to_csv(out_dir + '/' + fn + '_tile_idx.csv', index=False)
    slide_idx_df = idx_df.copy()
    slide_idx_df = slide_idx_df[['Patient_ID', 'Slide_ID', 'Tumor', 'label']].drop_duplicates()

    slide_idx_df.to_csv(out_dir + '/' + fn + '_slide_idx.csv', index=False)


def class_weight(label, verbose=False):
    neg, pos = np.bincount(label)
    total = neg + pos
    
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    if verbose:
        print('Training examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
               total, pos, 100 * pos / total))
        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))
    
    return class_weight
    

def stratified_weights(df, stratify='Tumor', weighted='label', verbose=True):
    all_weights = {}
        for level in df[stratify].unique():
            if len(df.loc[df[stratify] == level][weighted].value_counts())==1:
                weights = {0:1e-4,1:1}
            else:
                weights = class_weight(df.loc[df[stratify] == level][weighted])
            all_weights[level] = weights
        
        if verbose:
            print(pd.DataFrame(all_weights))
        
        df['sample_weights'] = [all_weights[x][y] for x, y in zip(df[stratify], df[weighted])]
        
        return df


