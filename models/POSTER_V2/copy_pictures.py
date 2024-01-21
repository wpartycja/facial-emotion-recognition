import os
import shutil



directory = '../../../datasets/soup_expw_by_student1'
out_directory = '../../../datasets/soup_main_2'

for classy in os.listdir(directory):
    print("Start of class ", classy)
    class_path = f'{directory}/{classy}'
    for filename in os.listdir(class_path):
        fullpath = os.path.join(class_path, filename)
        shutil.copyfile(fullpath, f'{out_directory}/{classy}/expw_{filename}')
