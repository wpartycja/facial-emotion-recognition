import os
# from PIL import Image
import shutil

directory = '../../../datasets/affectnet_not_in_soup'
out_directory = '../../../datasets/soup_main_2'

affect_to_raf = {0: 6, 1: 3, 2: 4, 3: 0, 4: 1, 5: 2, 6: 5}

proportion = {0: 21103,
             1: 37885,
             2: 7176,
             3: 3971,
             4: 1798,
             5: 1072,
             6: 7013
            }

for classy in os.listdir(directory):
    print("Start of class ", classy)
    raf_class = affect_to_raf[int(classy)]
    class_path = f'{directory}/{classy}'
    i = 0
    for filename in os.listdir(class_path):
        fullpath = os.path.join(class_path, filename)
        shutil.copyfile(fullpath, f'{out_directory}/{raf_class}/aff_{filename}')
        i += 1
        if i == proportion[int(classy)]:
            break 