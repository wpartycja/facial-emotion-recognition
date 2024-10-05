import splitfolders

path = 'path/to/your/dataset'
output = 'path/to/your/dest_dir'

splitfolders.ratio(path, output, seed=1337, ratio=(0.8, 0.2, 0))