import os
import openslide
import pandas as pd
import sklearn.utils as sku
import numpy as np
import re
import glob

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--tree',type=str,default='/gpfs/data/proteomics/projects/Josh_imaging_TCGA/tiles/PAAD')
parser.add_argument('--output',type=str,default='PAAD_tiles.csv')
args = parser.parse_args()

tree = args.tree
potentialList = glob.glob(tree+'*/*/*/',recursive=True)

def paired_tile_ids_in(slide, label, root_dir, age=None, BMI=None):
    imageDir = '/'.join(root_dir.split('/')[:6])+'/images/'+'/'.join(root_dir.split('/')[7:9])+'/'
    images = os.listdir(imageDir)
    images = list(filter(lambda x:'.svs' in x, images))
    image = imageDir+list(filter(lambda x:root_dir.split('/')[-1] in x, images))[0]
    # print(image)
    
    if '.svs' in image:
        slide = openslide.OpenSlide(image)
        max = int(slide.properties['aperio.AppMag'])
        # print(max)
        
    dira = os.path.isdir(root_dir + 'level1')
    dirb = os.path.isdir(root_dir + 'level2')
    dirc = os.path.isdir(root_dir + 'level3')
    if dira and dirb and dirc:
        # if "TCGA" in root_dir:
        if max==40:
            fac = 2000
        else:
            fac = 1000
        ids = []
        for level in range(1, 4):
            dirr = root_dir + 'level{}'.format(str(level))
            for id in os.listdir(dirr):
                if '.png' in id:
                    x = int(float(id.split('x-', 1)[1].split('-', 1)[0]) / fac)
                    y = int(float(re.split('_', id.split('y-', 1)[1])[0][:-4]) / fac)
                    try:
                        dup = re.split('.p', re.split('_', id.split('y-', 1)[1])[1])[0]
                        
                    except IndexError:
                        dup = np.nan
                    ids.append([slide, label, level, dirr + '/' + id, x, y, dup])
                else:
                    pass
                    # print('Skipping ID:', id)
        ids = pd.DataFrame(ids, columns=['slide', 'label', 'level', 'path', 'x', 'y', 'dup'])
        idsa = ids.loc[ids['level'] == 1]
        idsa = idsa.drop(columns=['level'])
        idsa = idsa.rename(index=str, columns={"path": "L0path"})
        idsb = ids.loc[ids['level'] == 2]
        idsb = idsb.drop(columns=['slide', 'label', 'level'])
        idsb = idsb.rename(index=str, columns={"path": "L1path"})
        idsc = ids.loc[ids['level'] == 3]
        idsc = idsc.drop(columns=['slide', 'label', 'level'])
        idsc = idsc.rename(index=str, columns={"path": "L2path"})
        idsa = pd.merge(idsa, idsb, on=['x', 'y', 'dup'], how='left', validate="many_to_many")
        idsa['x'] = idsa['x'] - (idsa['x'] % 2)
        idsa['y'] = idsa['y'] - (idsa['y'] % 2)
        idsa = pd.merge(idsa, idsc, on=['x', 'y', 'dup'], how='left', validate="many_to_many")
        idsa = idsa.drop(columns=['x', 'y', 'dup'])
        idsa = idsa.dropna()
        idsa = sku.shuffle(idsa)
        idsa['age'] = age
        idsa['BMI'] = BMI
        idsa['max'] = max
    else:
        idsa = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI','max'])
    return idsa



output = pd.DataFrame(columns=['patientIDs','imageIDs','cancerType','NT','L0path','L1path','L2path'])
counter = 1
appended_data = []
# for imageID in input['imageIDs'].unique():
for path in potentialList:
    print(path)
    print(counter/len(potentialList))
    counter = counter + 1
    root_dir = path
    slide = path.split('/')[-3]
    label = path.split('/')[-4]
    idsa = paired_tile_ids_in(slide=slide, label=label, root_dir=root_dir, age=None, BMI=None)
    idsa = idsa.drop(['age','BMI'],'columns')
    idsa['patientIDs'] = slide
    idsa['imageIDs'] = root_dir.split('/')[-2]
    idsa['label'] = label
    idsa['cancerType'] = root_dir.split('/')[-4]
    idsa['NT'] = slide.split('-')[-1]
    idsa = idsa[['patientIDs','imageIDs','cancerType','NT','L0path','L1path','L2path','max']]
    appended_data.append(idsa)

output = pd.concat(appended_data)
output.to_csv(args.output,index=False)