import sys
import os
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image


def get_images_and_masks(classes_id, min_pixels, partition, path_annot, mode=""):
    df = pd.read_csv(osp.join(path_annot, 'validation', 'validation-annotations-object-segmentation.csv'))
    kept = df.loc[df['LabelName'].isin(classes_id)]
    mv = [osp.join('validation/masks', e) for e in kept['MaskPath'].tolist()]
    iv = [osp.join('validation', e+'.jpg') for e in kept['ImageID'].tolist()]

    df = pd.read_csv(osp.join(path_annot, 'test', 'test-annotations-object-segmentation.csv'))
    kept = df.loc[df['LabelName'].isin(classes_id)]
    mte = [osp.join('test/masks', e) for e in kept['MaskPath'].tolist()]
    ite = [osp.join('test', e+'.jpg') for e in kept['ImageID'].tolist()]

    maskpaths = mv + mte
    impaths = iv + ite

    if partition != "val" and mode != "lite":
        df = pd.read_csv(osp.join(path_annot, 'train', 'train-annotations-object-segmentation.csv'))
        kept = df.loc[df['LabelName'].isin(classes_id)]
        mtr = [osp.join('train/masks', e) for e in kept['MaskPath'].tolist()]
        itr = [osp.join('train', e+'.jpg') for e in kept['ImageID'].tolist()]

        maskpaths += mtr
        impaths += itr

    if mode == "default":
        mode_flag = ""
    else:
        mode_flag = "_" + mode

    masksfile = "{}_masks{}.txt".format(partition, mode_flag)
    imagesfile = "{}_images{}.txt".format(partition, mode_flag)

    fm = open(osp.join(path_annot, masksfile), 'w')
    fi = open(osp.join(path_annot, imagesfile), 'w')
    for i in tqdm(range(len(maskpaths))):
        m = maskpaths[i]
        im = impaths[i]
        mask = np.array(Image.open(osp.join(path_annot, m)))

        pos = np.where(mask)
        if len(pos[0]) == 0:
            continue
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        w = xmax - xmin
        h = ymax - ymin
        if np.min((h, w)) < min_pixels or h*w < min_pixels*80:
            continue
        fm.write('{}\n'.format(m))
        fi.write('{}\n'.format(im))
    fm.close()
    fi.close()


if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    mode = ""

if mode == "lite":
    min_pixels = 80
else:
    min_pixels = 50


root = osp.join(os.environ['DB_ROOT'], 'OpenImages')
path_images = osp.join(root, 'images')
path_annot = osp.join(root, 'annotations')

all_classes = pd.read_csv(osp.join(root, 'metadata/class-descriptions-boxable.csv'), names=['ID', 'name'])

val_classes = ['Lion', 'Duck', 'Bus', 'Croissant', 'Panda', 'Parrot', 'Hammer', 'Airplane', 'Apple', 'Ambulance', 'Harmonica', 'Washing machine', 'Lighthouse', 'Hot dog', 'Hamburger']

val_classes_id = set(all_classes.loc[all_classes['name'].isin(val_classes)]['ID'].tolist())
train_classes_id = set(all_classes.loc[~all_classes['name'].isin(val_classes)]['ID'].tolist())

get_images_and_masks(val_classes_id, min_pixels, 'val', path_annot, mode)
get_images_and_masks(train_classes_id, min_pixels, 'train', path_annot, mode)
