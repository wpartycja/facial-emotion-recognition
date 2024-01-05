import os
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets

from models.make_target_model import make_target_model
import time

class Config:
    pass
cfg = Config()
cfg.ori_shape = (256, 256)
cfg.image_crop_size = (224, 224)
cfg.normalize_mean = [0.5, 0.5, 0.5]
cfg.normalize_std = [0.5, 0.5, 0.5]
cfg.last_stride = 2
cfg.num_classes = 8
cfg.num_branches = cfg.num_classes + 1
cfg.backbone = 'resnet18' # 'resnet18', 'resnet50_ibn'
# cfg.pretrained = "./weights/AffectNet_res18_acc0.6285.pth"
cfg.pretrained = "./weights/ExpW_res18_acc.pth"
cfg.pretrained_choice = '' # '' or 'convert'
cfg.bnneck = True  
cfg.BiasInCls = False


def create_loader(transform):

    valset = torchvision.datasets.ImageFolder('./ExpW-mini/valid', transform)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True
    )
    return val_loader
    # classes = valset.find_classes('./ExpW-mini/valid')[0]


def evaluate(device, val_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        running_loss = 0.0
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            # calculate outputs by running images through the network 
            outputs = model(images) # .cpu()
            # loss = criterion(outputs, labels)

            # runningloss += loss.item()

            # the class with the highest energy is what we choose as prediction
            probs = [F.softmax(pred, dim=-1) for pred in outputs]
            idxs  = [torch.argmax(prob.cpu()).item() for prob in probs]
            print(idxs)
            print(labels)
            for idx, label in zip(idxs, labels):
                if idx == label:
                    correct += 1
            # , predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # correct += (idxs == labels).sum().item()

    print(f'Accuracy: {round(100*correct/total, 2)}')
    return round(100*correct/total, 2)


if __name__ == '__main__':
    img_path = './images/test1.jpg'
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
    model = model.to(device)
    print('Loaded pretrained model from {0}'.format(cfg.pretrained))

    loader = create_loader(transform)
    result = evaluate(device, loader)


    # start = time.time()
    # inference(model, img_path, transform, is_cuda=True)
    # end = time.time()
    # print(end - start)
    