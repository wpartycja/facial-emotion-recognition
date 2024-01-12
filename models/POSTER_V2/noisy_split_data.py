import splitfolders
import os

path = "/home/fer/emotions/datasets/soup_main"
splitfolders.ratio(path,seed=1337, output="Soup_splitted", ratio=(0.8, 0.2, 0))