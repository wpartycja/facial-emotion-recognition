from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
from networks.dan import DAN
import time
import argparse

import matplotlib.pyplot as plt


class Model():
    def __init__(self, path = './checkpoints/affecnet8_epoch5_acc0.6209.pth' ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

        self.model = DAN(num_head=4, num_class=7, pretrained=False)
        recorder = RecorderMeter1(40)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0),cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)

        return faces

    def fer(self, img0):

        # img0 = Image.open(path).convert('RGB')

        faces = self.detect(img0)

        if len(faces) == 0:
            return 'null'

        ##  single face detection
        x, y, w, h = faces[0]

        
        img = img0.crop((x,y, x+w, y+h))

        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out, _, _ = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return label

if __name__ == "__main__":
    # args = parse_args()

    model = Model()
    path = 'test_images/karol.jpg'
    image = Image.open(path).convert('RGB')

    # image = args.image

    # assert os.path.exists(image), "Failed to load image file."
    start = time.time()
    label = model.fer(image)
    end = time.time()
    print(end-start)

    print(f'emotion label: {label}')


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

