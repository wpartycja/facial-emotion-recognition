import os
import sys
import glob
from tqdm import tqdm
import argparse

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets

from networks.dan import DAN

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

import datetime
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")
eps = sys.float_info.epsilon

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default='datasets/AfectNet/', help='AfectNet dataset path.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=8, help='Number of class.')
    parser.add_argument('--resume', type=str, default=None, metavar='PATH', help='path to checkpoint')


    return parser.parse_args()


class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=8, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class PartitionLoss(nn.Module):
    def __init__(self, ):
        super(PartitionLoss, self).__init__()
    
    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            ## add eps to avoid empty var case
            loss = torch.log(1+num_head/(var+eps))
        else:
            loss = 0
            
        return loss


class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

        # self.weights = self.weights.clamp(min=1e-5)

    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.imgs[i][1] for i in dataset.indices]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def run_training():
    args = parse_args()
    print("start")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    
    model = DAN(num_class=args.num_class, num_head=args.num_head)
    criterion_cls = torch.nn.CrossEntropyLoss().to(device)
    criterion_af = AffinityLoss(device, num_class=args.num_class)
    criterion_pt = PartitionLoss()
    params = list(model.parameters()) + list(criterion_af.parameters())
    optimizer = torch.optim.Adam(params,args.lr,weight_decay = 0)
    recorder = RecorderMeter(args.epochs)
    recorder1 = RecorderMeter1(args.epochs)

    best_acc = 0
    model.to(device)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            start_epoch = checkpoint['iter']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(checkpoint['best_accuracy'])
            best_acc = checkpoint['best_accuracy']
            recorder = checkpoint['recorder']
            recorder1 = checkpoint['recorder1']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['iter']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        
        start_epoch = 0


        
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
        ])
    

    print("making train dataset")
    train_dataset = datasets.ImageFolder(f'../../../datasets/ExpW_ready/train', transform = data_transforms)   # loading statically
    print("finished")

    print('Whole train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                            #    sampler=ImbalancedDatasetSampler(train_dataset),
                                               shuffle = False, 
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])      
                                                                      

    print("making val dataset")
    val_dataset = datasets.ImageFolder(f'../../../datasets/ExpW_ready/valid', transform = data_transforms_val)    # loading statically
    print("finished")

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)


    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.6)

    print("start")
    
    print(len(train_loader))
    for epoch in tqdm(range(start_epoch, args.epochs)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        print('hello')
        model.train()

        for idx, (imgs, targets) in enumerate(train_loader):

            # print(idx, end=' ')
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)
            
            out,feat,heads = model(imgs)

            loss = criterion_cls(out,targets) + criterion_af(feat,targets) + criterion_pt(heads)

            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt

        # message = f"[Epoch {epoch}] Training accuracy: {round(acc, 3)}. Loss: {running_loss}. LR: {optimizer.param_groups[0]['lr']}"
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
            f.write('\n')

        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
        
        with torch.no_grad():
            running_loss_val = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            for imgs, targets in val_loader:
        
                imgs = imgs.to(device)
                targets = targets.to(device)
                out,feat,heads = model(imgs)

                loss = criterion_cls(out,targets) + criterion_af(feat,targets) + criterion_pt(heads)

                running_loss_val += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                
            running_loss_val = running_loss_val/iter_cnt   
            scheduler.step()

            acc_val = bingo_cnt.float()/float(sample_cnt)
            acc_val = np.around(acc_val.numpy(),4)
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, acc_val, running_loss_val))
            
            # message = f'[Epoch {epoch}] Validation accuracy: {round(acc_val, 3)}. Loss: {running_loss_val}.'
            txt_name = './log/' + time_str + 'log.txt'
            with open(txt_name, 'a') as f:
                f.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, acc_val, running_loss_val))
                f.write('\n')
            
            recorder.update(epoch, running_loss, acc, running_loss_val, acc_val)
            # recorder1.update(output, target)
            curve_name = time_str + 'cnn.png'
            recorder.plot_curve(os.path.join('./log/', curve_name))

            

            if acc_val > best_acc:
                print("saving best model")
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_accuracy' : best_acc,
                            'recorder1': recorder1,
                            'recorder': recorder},
                            os.path.join('checkpoints', f"{time_str}_best.pth"))
                tqdm.write('Best Model saved!')
            
            print("saving model")
            torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_accuracy' : best_acc,
                            'recorder1': recorder1,
                            'recorder': recorder},
                            os.path.join('checkpoints', f"{time_str}.pth"))
            tqdm.write('Model saved')
            best_acc = max(acc_val,best_acc)
            tqdm.write("best_acc:" + str(best_acc))




class RecorderMeter(object):
    """From POSTER++ code: https://github.com/Talented-Q/POSTER_V2/blob/main/main.py#L458"""
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc * 100
        self.epoch_accuracy[idx, 1] = val_acc * 100
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)

class RecorderMeter1(object):
    """From POSTER++ code: https://github.com/Talented-Q/POSTER_V2/blob/main/main.py#L458"""
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, output, target):
        self.y_pred = output
        self.y_true = target

    def plot_confusion_matrix(self, cm, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        y_true = self.y_true
        y_pred = self.y_pred

        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        cm = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 8), dpi=120)

        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm_normalized[y_val][x_val]
            if c > 0.01:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
        # offset the tick
        tick_marks = np.arange(len(7))
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)

        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
        # show confusion matrix
        plt.savefig('./log/confusion_matrix.png', format='png')
        # fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print('Saved figure')
        plt.show()

    def matrix(self):
        target = self.y_true
        output = self.y_pred
        im_re_label = np.array(target)
        im_pre_label = np.array(output)
        y_ture = im_re_label.flatten()
        # im_re_label.transpose()
        y_pred = im_pre_label.flatten()
        im_pre_label.transpose()



if __name__ == "__main__":                    
    run_training()