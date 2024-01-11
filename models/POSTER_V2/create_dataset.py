import torch
import argparse
import warnings
import datetime
import os
from models.PosterV2_7cls import *
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsampler import ImbalancedDatasetSampler
import torchvision
from main import *

warnings.filterwarnings("ignore", category=UserWarning)

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
parser.add_argument('--save', type=str, default=r'../../../datasets/soup/')
parser.add_argument('--data', type=str, default=r'../../../datasets/ExpW_unlabeled')
# parser.add_argument('--model', type=str, default=r'./checkpoint/raf-db-model_best.pth')
parser.add_argument('--data_type', default='RAF-DB', choices=['RAF-DB', 'AffectNet-7', 'ExpW'],
                        type=str, help='dataset option')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model.pth')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model_best.pth')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=55, type=int, metavar='N')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')

parser.add_argument('--lr', '--learning-rate', default=0.000035, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=30, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=None, type=str, help='evaluate model on test set')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()



if __name__ == "__main__":

    # create model
    model = pyramid_trans_expr2(img_size=224, num_classes=7)

    model = torch.nn.DataParallel(model).cuda()

    model_PATH = './checkpoint/raf-db-model_best.pth'


    if os.path.isfile(model_PATH):
        print("=> loading checkpoint '{}'".format(model_PATH))
        checkpoint = torch.load(model_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(model_PATH, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(model_PATH))
        

    imagedir = args.data

    model.eval()

    image_dataset = datasets.ImageFolder(imagedir,
                                        transforms.Compose([transforms.Resize((224, 224)),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225]),
                                                            transforms.RandomErasing(p=1, scale=(0.05, 0.05))]))

    image_loader = torch.utils.data.DataLoader(image_dataset,
                                                sampler=ImbalancedDatasetSampler(image_dataset),
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)

    with torch.no_grad():
        c = 0
        for data in image_loader:

            images, labels = data
            images = images.cuda()

            # calculate outputs by running images through the network 
            outputs = model(images) 

            y_pred_test = torch.argmax(outputs, dim=1).squeeze().tolist()
            
            for image, label in zip(images, y_pred_test):
                print(image)
                print(label)

                torchvision.utils.save_image(image, args.save + f'{c}.png')
                c += 1
            


        
        # return y_pred_list, y_test_list_final