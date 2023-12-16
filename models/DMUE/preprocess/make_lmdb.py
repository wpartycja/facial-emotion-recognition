import os
import cv2
import lmdb
import skimage.io as io
from tqdm import tqdm


lmdb_output = "./AffectNet_lmdb/"
lb_txt = "msra_train_file.txt"
ori_root = "../../../datasets/ExpW-mini/train"
size = (256, 256)


lines = open(lb_txt, 'r').readlines()
env_w = lmdb.open(lmdb_output, map_size=600_000_000)
txn_w = env_w.begin(write=True)

for i in tqdm(range(len(lines))):
    k, label = lines[i].split(' ')
    img_path = k.split('/')[1]
    print(img_path)

    img = io.imread(os.path.join(ori_root, label[:-1],img_path))[:,:,:3]
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

    txn_w.put(key=k.encode('utf-8'), value=img.tobytes())

txn_w.commit()
env_w.close()
