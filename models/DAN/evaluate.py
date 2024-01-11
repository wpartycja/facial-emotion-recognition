import torch
import torchvision.transforms as T
import torchvision.datasets
from model import Model
import argparse
from networks.dan import DAN
from sklearn.metrics import classification_report
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Image file for evaluation.')
parser.add_argument('--evaluate', type=str, help='path to model checkpoint')
parser.add_argument('--data_type', type=str, help='Type of data, AffectNet-7, RAF-DB, ExpW',  choices=['RAF-DB', 'AffectNet-7', 'ExpW'])
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--data', type=str, help='path to dataset')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')

args = parser.parse_args()


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



def create_loader(transform, args):
    valset = torchvision.datasets.ImageFolder(args.data+'/valid', transform)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    return val_loader
    # classes = valset.find_classes('./ExpW-mini/valid')[0]


def evaluate(model, device, val_loader):
    y_pred_list = []
    y_test_list = []

    model.eval()


    if args.data_type == 'AffectNet':
        target_names = ["neutral", 'happy', 'sad', 'surprise', 'fear', 'disgust', 'angry']
    elif args.data_type == 'ExpW':
        target_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    else:
        target_names = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']


    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network 
            outputs, _, _ = model(images)
            y_pred_test = torch.argmax(outputs, dim=1)
            # print(y_pred_test).squeeze().tolist()
            y_pred_list.extend(y_pred_test.squeeze().tolist())
            y_test_list.extend(labels.squeeze().tolist())
            # print('Pred: ', y_pred_list)
            # print('Labels: ', y_test_list)

    expw_to_affect = {0: 6, 1: 5, 2: 4, 3: 1, 4: 2, 5: 3, 6: 0}
    affect_to_expw = {v: k for k, v in expw_to_affect.items()}

    expw_to_raf = {0: 5, 1: 2, 2: 1, 3: 3, 4: 4, 5: 0, 6: 6}
    raf_to_expw = {v: k for k, v in expw_to_raf.items()}

    affect_to_raf = {0: 6, 1: 3, 2: 4, 3: 0, 4: 1, 5: 2, 6: 5}
    raf_to_affect = {v: k for k, v in affect_to_raf.items()}


    y_test_list_final = y_test_list
    y_test_list_final = [raf_to_expw[elem] for elem in y_test_list] # tu zmieniamy labelki z datasetowych na modelowe

    conf_matrix = confusion_matrix(y_pred_list, y_test_list_final, labels=[0,1,2,3,4,5,6] )
    class_report = classification_report(y_pred_list, y_test_list_final, target_names = target_names) 
    print(conf_matrix)
    print(class_report)

    file_name = './evaluate/' + args.evaluate.split('/')[-1].split('.')[0] + '_' + args.data.split('/')[-1] + '_log'
    txt_name = file_name + '.txt'
    png_name = file_name + '.png'

    ConfusionMatrixDisplay.from_predictions(y_test_list_final, y_pred_list, cmap='RdPu')
    plt.show()
    plt.savefig(png_name)
    
    with open(txt_name, 'w+') as f:
        f.write(class_report)
        f.write(str(conf_matrix))

    
            
            # _, preds = torch.max(outs,1)
            # idxs = [int(pred) for pred in preds]

            # print(idxs)
            # print(labels)
            # for idx, label in zip(idxs, labels):
            #     if idx == label:
            #         correct += 1
            # total += labels.size(0)

    # print(f'Accuracy: {round(100*correct/total, 2)}')
    # return round(100*correct/total, 2)


if __name__ == '__main__':
    device = 'cuda'

    transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomApply([
                T.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),

        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        T.RandomErasing(),
        ])

    print('Building model......')
    # model = Model(args.evaluate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    
    model = DAN(num_class=7, num_head=4)
    criterion_cls = torch.nn.CrossEntropyLoss().to(device)
    criterion_af = AffinityLoss(device, num_class=7)
    criterion_pt = PartitionLoss()
    params = list(model.parameters()) + list(criterion_af.parameters())
    optimizer = torch.optim.Adam(params,args.lr,weight_decay = 0)
    recorder = RecorderMeter(40)
    recorder1 = RecorderMeter1(40)

    print("=> loading checkpoint '{}'".format(args.evaluate))
    checkpoint = torch.load(args.evaluate, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    start_epoch = checkpoint['iter']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(checkpoint['best_accuracy'])
    best_acc = checkpoint['best_accuracy']
    recorder = checkpoint['recorder']
    recorder1 = checkpoint['recorder1']
    print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['iter']))
       
    print('Loaded pretrained model')
    model.to(device)
    loader = create_loader(transform, args)
    evaluate(model, device, loader)


