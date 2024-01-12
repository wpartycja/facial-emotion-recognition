import os
import shutil



directory = '../../../datasets/AffectNet_midi_labele_jak_w_rafie'
out_directory = '../../../datasets/soup_expw_plus_raf'

for classy in os.listdir(directory):
    print("Start of class ", classy)
    class_path = f'{directory}/{classy}'
    for filename in os.listdir(class_path):
        fullpath = os.path.join(class_path, filename)
        shutil.copyfile(fullpath, f'{out_directory}/{classy}/{filename}.png')
