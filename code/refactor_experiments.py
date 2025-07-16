import numpy as np
import torch 
import random
import os
from pathlib import Path
from sklearn.model_selection import KFold
from fastai.vision.all import *
import sys 
sys.path.append('..')

from refactor_dataset import get_segmentation, random_crop_tfm, PadToMultiple, CutInHalfTransform, load_mask_with_ignore
from config import *

def random_seed(seed_value, use_cuda = True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

trainFolders = [
     'ARMS',
     'AisazuNihaIrarenai',
     'AkkeraKanjinchou',
     'Akuhamu',
     'AosugiruHaru',
     'AppareKappore',
     'Arisa',
     'BEMADER_P',
     'BakuretsuKungFuGirl',
     'Belmondo',
     'BokuHaSitatakaKun',
     'BurariTessenTorimonocho',
     'ByebyeC-BOY',
     'Count3DeKimeteAgeru',
     'DollGun',
     'Donburakokko',
     'DualJustice',
     'EienNoWith',
     'EvaLady',
     'EverydayOsakanaChan',
     'GOOD_KISS_Ver2',
     'GakuenNoise',
     'GarakutayaManta',
     'GinNoChimera',
     'Hamlet',
     'HanzaiKousyouninMinegishiEitarou',
     'HaruichibanNoFukukoro',
     'HarukaRefrain',
     'HealingPlanet',
     "UchiNoNyan'sDiary",
     'UchuKigekiM774',
     'UltraEleven',
     'UnbalanceTokyo',
     'WarewareHaOniDearu',
     'YamatoNoHane',
     'YasasiiAkuma',
     'YouchienBoueigumi',
     'YoumaKourin',
     'YukiNoFuruMachi',
     'YumeNoKayoiji',
     'YumeiroCooking',
     'TotteokiNoABC',
     'ToutaMairimasu',
     'TouyouKidan',
     'TsubasaNoKioku'
]

assert(len(trainFolders) == len(set(trainFolders)))   
for x in trainFolders:
    # print(MASKS_PATH + '/' + x)
    # print(os.path.isdir('..'+MASKS_PATH + '/' + x))
    assert(os.path.isdir(MASKS_PATH + '/' + x))

def getDatasetLists(dataset):
    return [dataset.train.x, dataset.train.y, dataset.valid.x, dataset.valid.y]

def getData():
    """
    Returns a list of valid image files in MANGA109_PATH that have a corresponding mask and are in trainFolders.
    """
    random_seed(42)
    items = get_image_files(MANGA109_PATH)
    valid_items = [x for x in items if Path(get_segmentation(x)).exists() and x.parent.name in trainFolders]
    return valid_items

def getDatasets(all_files, crop=True, padding=8, cutInHalf=True, n_splits=5):
    """
    Returns a list of DataLoaders for KFold splits using fastai v2 DataBlock API.
    """
    datasets = []
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=42).split(trainFolders)
    folder_to_items = {folder: [f for f in all_files if Path(f).parent.name == folder] for folder in trainFolders}

    for _, valid_idx in folds:
        valid_folders = set([trainFolders[i] for i in valid_idx])
        train_items, valid_items = [], []
        for folder, files in folder_to_items.items():
            (valid_items if folder in valid_folders else train_items).extend(files)

        def label_func(fn):
            return get_segmentation(fn)

        def open_mask_with_ignore(fn):
            return load_mask_with_ignore(fn)

        dblock = DataBlock(
            blocks=[ImageBlock, MaskBlock(codes=['background', 'text'])],
            get_items=noop,
            splitter=FuncSplitter(lambda o: o in valid_items),
            get_y=label_func,
            item_tfms=[Resize((800, 800), method='squish'), PadToMultiple(8), CutInHalfTransform(enabled=cutInHalf)],
            batch_tfms=[random_crop_tfm() if crop else noop, IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)]
        )
        # print("train_items:", train_items)
        # print("valid_items:", valid_items)
        # print("all items:", train_items + valid_items)
        # for fn in train_items + valid_items:
        #     label = get_segmentation(fn)
        #     if not Path(label).exists():
        #         print(f"Label does not exist for {fn}: {label}")
        dls = dblock.dataloaders(train_items + valid_items, bs=4)
        datasets.append(dls)
    return datasets

def getDataset(allData):
    """
    Returns a single dataset for non-manga data, with all items in validation.
    """
    dataset = (allData
        .split_by_valid_func(lambda _: True)
        .label_from_func(get_segmentation, classes=['text']))
    for l in getDatasetLists(dataset):
        l.padding = 8
        l.cutInHalf = False
    return dataset

def colorizePrediction(prediction, truth):
    """
    Given prediction and ground truth, returns colorized tensor with true positives as green, false positives as red, and false negatives as white.
    """
    prediction, truth = prediction[0], truth[0]
    colorized = torch.zeros(4, prediction.shape[0], prediction.shape[1]).int()
    r, g, b, a = colorized[:]
    fn = (truth >= 1) & (truth <= 5) & (truth != 3) & (prediction == 0)
    tp = ((truth >= 1) & (truth <= 5)) & (prediction == 1)
    fp = (truth == 0) & (prediction == 1)
    r[fp] = 255
    r[fn] = g[fn] = b[fn] = 255
    g[tp] = 255
    a[:, :] = 128
    a[tp | fn | fp] = 255
    return colorized

def findImage(datasets, folder, index):
    """
    Given an image index from a folder (like ARMS, 0), finds which dataset has it in validation and the index inside it.
    """
    for dIndex, dataset in enumerate(datasets):
        for idx, item in enumerate(dataset.valid.y.items):
            if "/" + folder + "/" in str(item) and str(index) + ".png" in str(item):
                return dIndex, idx