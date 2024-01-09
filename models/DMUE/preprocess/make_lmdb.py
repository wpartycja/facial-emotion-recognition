import os
import cv2
import lmdb
import skimage.io as io
from tqdm import tqdm


lmdb_output = "../ExpW_lmdb"
lb_txt = "../lb2.txt"
ori_root = "/home/fer/emotions/datasets/ExpW_ready"
size = (256, 256)

lines = open(lb_txt, 'r').readlines()
env_w = lmdb.open(lmdb_output, writemap=True, map_size=50_000_000_000)

with env_w.begin(write=True) as txn_w:
    for i in tqdm(range(len(lines))):
        k = lines[i].split(' ')[0]
        img_path =  '_'.join(k.split('_')[1:]) # hit jakis

        img = io.imread(os.path.join(ori_root, img_path))[:,:,:3]
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        txn_w.put(key=k.encode('utf-8'), value=img.tobytes())

    # txn_w.commit()

