import os

directory = '/home/fer/emotions/datasets/ExpW-mini'
file = './lb2.txt'

labels = list()

for dir in os.listdir(directory):
    for label in os.listdir(f'{directory}/{dir}'):
        for image in os.listdir(f'{directory}/{dir}/{label}'):
            labels.append(f"{dir}_{dir}/{label}/{image} {label}")

with open(file, 'w+') as f:
    f.write('\n'.join(labels))