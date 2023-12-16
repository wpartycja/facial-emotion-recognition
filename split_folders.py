import splitfolders

path = '../datasets/ExpW_clean'
output = '../datasets/ExpW_ready'

splitfolders.ratio(path, output, seed=1337, ratio=(.8, 0.2, 0))