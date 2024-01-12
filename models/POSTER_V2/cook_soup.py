import shutil
import warnings
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
warnings.filterwarnings("ignore")
import torch.utils.data as data
import os
import argparse
from sklearn.metrics import f1_score, confusion_matrix
from data_preprocessing.sam import SAM
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
from torchsampler import ImbalancedDatasetSampler
from models.PosterV2_7cls import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import time

warnings.filterwarnings("ignore", category=UserWarning)

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=r'/home/Dataset/RAF')
parser.add_argument('--data_type', default='RAF-DB', choices=['RAF-DB', 'AffectNet-7', 'ExpW'],
                        type=str, help='dataset option')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model.pth')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model_best.pth')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=144, type=int, metavar='N')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')

parser.add_argument('--lr', '--learning-rate', default=0.000035, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=30, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=None, type=str, help='evaluate model on test set')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gpu', type=str, default='0')

parser.add_argument('--model_type', default='RAF-DB', choices=['RAF-DB', 'AffectNet', 'ExpW'],
                        type=str, help='dataset option')

args = parser.parse_args()


def main():
    directory = '../../../datasets/ExpW_ready/train'
    out_directory = '../../../datasets/soup_expw'
    model_checkpoint = './checkpoint/raf-db-model_best.pth'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_acc = 0
    print('Start time: ' + now.strftime("%m-%d %H:%M"))

    # create model
    model = pyramid_trans_expr2(img_size=224, num_classes=7)

    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(model_checkpoint):
        print("=> loading checkpoint '{}'".format(model_checkpoint))
        checkpoint = torch.load(model_checkpoint)
        best_acc = checkpoint['best_acc']
        best_acc = best_acc.to()
        print(f'best_acc:{best_acc}')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(model_checkpoint, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_checkpoint))
        
        # validate(val_loader, model, criterion, args)
            
    data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225]),
                                                            ])
    model.eval()
    i = 0
    start_whole = time.time()

    # for dir in os.listdir(directory):
    #     for label in os.listdir(f'{directory}/{dir}'):
    #         for image in os.listdir(f'{directory}/{dir}/{label}'):

    for classy in os.listdir(directory):
        print("Start of class ", classy)
        start = time.time()
        class_path = f'{directory}/{classy}'
        for idx, filename in enumerate(os.listdir(class_path)):
            fullpath = os.path.join(class_path, filename)
            img0 = Image.open(fullpath).convert('RGB')
            img = data_transforms(img0)
            img = img.view(1,3,224,224)
            img = img.to(device)

            with torch.set_grad_enabled(False):
                out = model(img)
                # print('out: ', out)
                pred = torch.argmax(out,1)
                # print('pred: ', pred)
                index = int(pred)
                # print(index)

            img0.save(f'{out_directory}/{index}/{idx}.png')
            i+=1
        end = time.time()
        print(f"End of class {classy}, exec time: {end-start}, number of images: {idx}")
        print("")
    end_whole = time.time()
    print("Number of images: ", i)
    print("Exec time: ", end_whole-start_whole)


if __name__ == '__main__':
    main()



