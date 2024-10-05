import os

directory = '../../../datasets/ExpW-mini'
for dir in os.listdir(directory):
    for label in os.listdir(f'{directory}/{dir}'):
        for image in os.listdir(f'{directory}/{dir}/{label}'):
            print(f"{dir}_{dir}/{label}/{image} {label}")