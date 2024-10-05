import os
import json
import shutil

# @TODO: wez to jakos ladnie
train_path = '../../datasets/AffectNet/train_set/images'
val_path = '../../datasets/AffectNet/val_set/images'
dest = '../../datasets/AffectNet_clean'

def order_dataset(csv_path: str, json_path_train: str, json_path_val: str) -> None:
    """
    reading information from .csv file
    create .json for train and val set
    with label for every image
    """
    
    dict_train, dict_val = dict(), dict()

    with open(csv_path, 'r') as file:
        for line in file:
            line = line.split('/')
            data_type = line[0].split(',')[1]

            line = line[-1].split(',')
            line[1] = line[1][0]
            name, label = line

            # choose dataset types
            if data_type == 'train':
                dict_train[name] = label
            elif data_type == 'val':
                dict_val[name] = label
            
    # save to .json files
    with open(json_path_train, 'w') as file:
        json.dump(dict_train, file)
    
    with open(json_path_val, 'w') as file:
        json.dump(dict_val, file)  


def create_ordered_dataset(source_path: str, dest_path: str, dir_name: str, json_path: str) -> None:
    """
    taking information from generated .json files
    copies images to dedicated directories

    Args:
        json_path (str): _description_
        dir_name (str): _description_
    """
    with open(json_path, 'r') as fp:
        label_dict = json.load(fp)

    for file in os.listdir(source_path):
        label = label_dict[file]

        shutil.copyfile(f'{source_path}/{file}', f'{dest_path}/{dir_name}/{label}/{file}')


if __name__ == "__main__":
    order_dataset('./input/affectnet.csv', './output/affectnet_train.json', './output/affectnet_val.json')
    
    create_ordered_dataset(train_path, dest, 'train_set', './affectnet_train.json')
    create_ordered_dataset(val_path, dest, 'val_set', './affectnet_val.json')