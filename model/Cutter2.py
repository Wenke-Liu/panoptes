"""
Tile svs/scn files

Created on 11/01/2018

@author: RH
"""

import time
import os
import subprocess
import matplotlib
import os
import shutil
import pandas as pd
matplotlib.use('Agg')
import Slicer2
import staintools
import re
import openslide


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prefix',type=str,default='/home/epoch/josh/Josh_imaging_TCGA/images/UCEC/TCGA-2E-A9G8-tumor')
# parser.add_argument('--slideInput',type=str,default='/home/epoch/josh/Josh_imaging_TCGA/images/UCEC/TCGA-2E-A9G8-tumor')
parser.add_argument('--colorStandard',type=str,default='/gpfs/data/proteomics/projects/Josh_imaging_TCGA/colorStandard.png')
parser.add_argument('--tileOutput',type=str,default='/gpfs/data/proteomics/projects/mh6486/LUAD_Imaging/tiles')

args = parser.parse_args()
print(args)

# Get all images in the root directory
def image_ids_in(root_dir, mode, ignore=['.DS_Store', 'dict.csv', 'label.csv','manifest.txt','sample_sheet.tsv']):
    ids = []
    for id in os.listdir(root_dir):
        if '.svs' in id:
            if id in ignore:
                print('Skipping ID:', id)
            else:
                if mode == 'CPTAC':
                    # dirname = id.split('_')[-3]
                    # sldnum = id.split('_')[-2].split('-')[-1]
                    dirname = 'C3L-02557-23'
                    sldnum = '23'
                    ids.append((id, dirname, sldnum))
                if mode == 'TCGA':
                    # dirname = re.split('-01Z|-02Z|-01A', id)[0]
                    dirname = root_dir.split('/')[-2]
                    sldnum = id.split('-')[5].split('.')[0]
                    ids.append((id, dirname, sldnum))
    return ids


# cut; each level is 2 times difference (20x, 10x, 5x)
def cut(slideInput,tileOutput,colorStandard,CPTACpath='',TCGApath='',slideFile=''):
    # load standard image for normalization
    std = staintools.read_image(colorStandard)
    std = staintools.LuminosityStandardizer.standardize(std)
    CPTACpath = CPTACpath
    TCGApath = TCGApath
    # ref = pd.read_csv(slideFile, header=0)
    # refls = ref['name'].tolist() ##what is this?
    # # cut tiles with coordinates in the name (exclude white)
    start_time = time.time()
    
    if(CPTACpath!=''):
        CPTACpath = args.prefix+'/'
        CPTAClist = image_ids_in(CPTACpath, 'CPTAC')
        CPTACpp = pd.DataFrame(CPTAClist, columns=['id', 'dir', 'sld'])
        CPTACcc = CPTACpp['dir'].value_counts()
        CPTACcc = CPTACcc[CPTACcc > 1].index.tolist()
        print(CPTACcc)
    else:
        CPTAClist = []
    if(TCGApath!=''):
        TCGApath = args.prefix+'/'
        TCGAlist = image_ids_in(TCGApath, 'TCGA')
        TCGApp = pd.DataFrame(TCGAlist, columns=['id', 'dir', 'sld'])
        TCGAcc = TCGApp['dir'].value_counts()
        TCGAcc = TCGAcc[TCGAcc > 1].index.tolist()
        print(TCGAcc)
    else:
        TCGAlist = []

    # CPTAC
    for i in CPTAClist:
        for m in [1,2,3]:
            if m == 0:
                tff = 1
                level = 0
            elif m == 1:
                tff = 2
                level = 0
            elif m == 2:
                tff = 1
                level = 1
            elif m == 3:
                tff = 2
                level = 1
                
            otdir = args.prefix+'/tiles/level'+str(m)
            try:
                os.makedirs(otdir)
            except(FileExistsError):
                pass
            try:
                print(args.prefix+'/'+i[0])
                n_x, n_y, raw_img, ct = Slicer2.tile(image_file=args.prefix+'/'+i[0], outdir=otdir,
                                                                level=level, std_img=std, ft=tff)
            except(IndexError):
                pass
            if len(os.listdir(otdir)) < 2:
                shutil.rmtree(otdir, ignore_errors=True)
        # else:
        #     print("pass: {}".format(str(i)))

    # TCGA
    for i in TCGAlist:
        # matchrow = ref.loc[ref['name'] == i[1]]
        # if matchrow.empty:
        #     continue
        print(i)
        
        slide = openslide.OpenSlide(args.prefix+'/'+i[0])
        max = int(slide.properties['aperio.AppMag'])

        try:
            os.mkdir(tileOutput+"/{}".format(i[1]))
            dir = tileOutput+"/{}/".format(i[1])
            os.system('chmod 777 -R '+dir)        

            # os.mkdir("../tiles/{}".format(i[1]))
        except(FileExistsError):
            pass

        try:
            os.mkdir(tileOutput+"/{}/{}".format(i[1],i[2]))
        except(FileExistsError):
            pass

        if max==40:
            print('max is 40x')
            for m in [1,2,3]: #range 4
                if m == 0:
                    tff = 2
                    level = 0
                elif m == 1:
                    tff = 1
                    level = 1
                elif m == 2:
                    tff = 2
                    level = 1
                elif m == 3:
                    tff = 1
                    level = 2

                # otdir = tileOutput+"/{}/level{}/".format(i[2],str(m))
                otdir = tileOutput+"/{}/{}/level{}/".format(i[1],i[2],str(m))
                try:
                    os.mkdir(otdir)
                except(FileExistsError):
                    pass
                # try:

                n_x, n_y, raw_img, ct = Slicer2.tile(image_file=args.prefix+'/'+i[0], outdir=otdir,
                                                                    level=level, std_img=std, ft=tff)
                # except Exception as e:
                    # print('Error!')
                    # pass

                if len(os.listdir(otdir)) < 2:
                    shutil.rmtree(otdir, ignore_errors=True)
        elif max==20:
            print('max is 20x')
            for m in [1,2,3]:
                if m == 0:
                    tff = 1
                    level = 0
                elif m == 1:
                    tff = 2
                    level = 0
                elif m == 2:
                    tff = 1
                    level = 1
                elif m == 3:
                    tff = 2
                    level = 1
                    
                # otdir = tileOutput+"/{}/level{}/".format(i[2],str(m))
                otdir = tileOutput+"/{}/{}/level{}/".format(i[1],i[2],str(m))
                try:
                    os.mkdir(otdir)
                except(FileExistsError):
                    pass
                # try:

                n_x, n_y, raw_img, ct = Slicer2.tile(image_file=args.prefix+'/'+i[0], outdir=otdir,
                                                                    level=level, std_img=std, ft=tff)
                # except Exception as e:
                    # print('Error!')
                    # pass

                if len(os.listdir(otdir)) < 2:
                    shutil.rmtree(otdir, ignore_errors=True)


        dir = tileOutput+"/{}/{}/".format(i[1],i[2])
        os.system('chmod 777 -R '+dir)
        # for root, dirs, files in os.walk(dir):
        #     for d in dirs:
        #         os.chmod(os.path.join(root, d), 0o777)
        #     for f in files:
        #         os.chmod(os.path.join(root, f), 0o777)

    print("--- %s seconds ---" % (time.time() - start_time))
    # subfolders = [f.name for f in os.scandir(tileOutput) if f.is_dir()]
    # for w in subfolders:
    #     if w not in refls:
    #         print(w)
    # # Time measure tool
    # start_time = time.time()
    # print("--- %s seconds ---" % (time.time() - start_time))



# Run as main
if __name__ == "__main__":
    # if not os.path.isdir('../tiles'):
    #     os.mkdir('../tiles')

    cut(slideInput=args.prefix,tileOutput = args.tileOutput,colorStandard=args.colorStandard,TCGApath=args.prefix)

