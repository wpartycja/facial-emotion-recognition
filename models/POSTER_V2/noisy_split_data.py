import splitfolders
import os

path = "/home/fer/emotions/datasets/soup_main_2"
out_path = "/home/fer/emotions/datasets/soup_main_2_splitted"
splitfolders.ratio(path,seed=1337, output=out_path, ratio=(0.8, 0.2, 0))