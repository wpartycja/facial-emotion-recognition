import os
import json
import shutil



train_path = '../../datasets/AffectNet/train_set/images'
val_path = '../../datasets/AffectNet/val_set/images'
test_path = '../../AffectNet_test/samples/test/happy'
dest = '../../datasets/AffectNet_clean'


# my_dict = dict()


# with open('affectnet.csv', 'r') as file:
#     for line in file:
#         line = line.split('/')
#         data_type = line[0].split(',')[1]

#         if data_type == 'val':
#             line = line[-1].split(',')
#             line[1] = line[1][0]
#             name, label = line
#             my_dict[name] = label

# with open('./affectnet_val.json', 'w') as file:
#     json.dump(my_dict, file)


with open('./affectnet_train.json', 'r') as fp:
    label_dict = json.load(fp)

for file in os.listdir(train_path):
    label = label_dict[file]

    shutil.copyfile(f'{train_path}/{file}', f'{dest}/train_set/{label}/{file}')
