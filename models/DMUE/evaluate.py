import os
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets

from models.make_target_model import make_target_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Config:
    pass
cfg = Config()
cfg.ori_shape = (256, 256)
cfg.image_crop_size = (224, 224)
cfg.normalize_mean = [0.5, 0.5, 0.5]
cfg.normalize_std = [0.5, 0.5, 0.5]
cfg.last_stride = 2
cfg.num_classes = 7
cfg.num_branches = cfg.num_classes + 1
cfg.backbone = 'resnet18' # 'resnet18', 'resnet50_ibn'
cfg.pretrained_choice = '' # '' or 'convert'
cfg.bnneck = True  
cfg.BiasInCls = False

# TO CHANGE: :)
cfg.pretrained = "./weights/ExpW_now.pth"
cfg.data_path = '../../../datasets/ExpW_ready/valid'
model_types = ['ExpW', 'AffectNet', 'RAF-DB']
cfg.model_type = model_types[0]


def create_loader(transform):

    valset = torchvision.datasets.ImageFolder(cfg.data_path, transform)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    return val_loader
    # classes = valset.find_classes('./ExpW-mini/valid')[0]


def evaluate(device, val_loader):

    y_pred_list = []
    y_test_list = []

    if cfg.model_type == 'AffectNet':
        target_names = ["neutral", 'happy', 'sad', 'surprise', 'fear', 'disgust', 'angry']
    elif cfg.model_type == 'ExpW':
        target_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    else:
        target_names = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']


    with torch.no_grad():
        print(len(val_loader))
        for data in val_loader:

            images, labels = data
            images = images.to(device)
            labels = labels.to(device)


            # calculate outputs by running images through the network 
            outputs = model(images) # .cpu()

            y_pred_test = torch.argmax(outputs, dim=1)
            y_pred_list.extend(y_pred_test.squeeze().tolist())
            y_test_list.extend(labels.squeeze().tolist())

    expw_to_affect = {0: 6, 1: 5, 2: 4, 3: 1, 4: 2, 5: 3, 6: 0}
    affect_to_expw = {v: k for k, v in expw_to_affect.items()}

    expw_to_raf = {0: 5, 1: 2, 2: 1, 3: 3, 4: 4, 5: 0, 6: 6}
    raf_to_expw = {v: k for k, v in expw_to_raf.items()}

    affect_to_raf = {0: 6, 1: 3, 2: 4, 3: 0, 4: 1, 5: 2, 6: 5}
    raf_to_affect = {v: k for k, v in affect_to_raf.items()}

    y_test_list_final = y_test_list
    # y_test_list_final = [affect_to_raf[elem] for elem in y_test_list] # tu zmieniamy labelki z datasetowych na modelowe


    conf_matrix = confusion_matrix(y_pred_list, y_test_list_final, labels=[0,1,2,3,4,5,6] )
    class_report = classification_report(y_pred_list, y_test_list_final, target_names = target_names) 
    print(conf_matrix)
    print(class_report)

    file_name = cfg.pretrained.split('/')[-1].split('.')[0] + '_' + cfg.data_path.split('/')[-1] + '_log'
    txt_name = './evaluate/' +  file_name + '.txt'
    png_name = './evaluate/confusion_matrices/' + file_name + '.png'
    png_name_n = './evaluate/confusion_matrices_normalized/' + file_name + '.png'


    ConfusionMatrixDisplay.from_predictions(y_test_list_final, y_pred_list, cmap='RdPu', display_labels=target_names)
    plt.savefig(png_name)
    ConfusionMatrixDisplay.from_predictions(y_test_list_final, y_pred_list, cmap='RdPu', display_labels=target_names, normalize='true')
    plt.savefig(png_name_n)
    
    with open(txt_name, 'w+') as f:
        f.write(class_report)
    


if __name__ == '__main__':
    device = 'cuda'

    transform = T.Compose([
            T.Resize(cfg.ori_shape),
            T.CenterCrop(cfg.image_crop_size),
            T.ToTensor(),
            T.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
        ])

    print('Building model......')
    model = make_target_model(cfg)
    model.load_param(cfg)
    model.eval()

    model = model.to(device)
    print('Loaded pretrained model from {0}'.format(cfg.pretrained))

    target_names = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}
    loader = create_loader(transform)
    evaluate(device, loader)



    # start = time.time()
    # inference(model, img_path, transform, is_cuda=True)
    # end = time.time()
    # print(end - start)
    