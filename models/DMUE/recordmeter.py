import matplotlib.pyplot as plt
import numpy as np

class RecorderMeter(object):
    """From POSTER++ code: https://github.com/Talented-Q/POSTER_V2/blob/main/main.py#L458"""
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 5), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 1), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc * 100
        self.epoch_accuracy[idx, 1] = val_acc * 100
        self.current_epoch = idx + 1
    

    def update_acc(self, idx, acc):
        self.epoch_accuracy[idx, 0] = acc * 100
    

    def update_losses(self, idx, softLoss, aux_loss, CEloss, spLoss, total_loss):
        self.epoch_losses[idx, 0] = softLoss * 2000
        self.epoch_losses[idx, 1] = aux_loss * 30
        self.epoch_losses[idx, 2] = CEloss * 30
        self.epoch_losses[idx, 3] = spLoss * 2000
        self.epoch_losses[idx, 4] = total_loss
        

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

        # y_axis[:] = self.epoch_accuracy[:, 1]
        # plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        # plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='softLoss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='aux_loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 2]
        plt.plot(x_axis, y_axis, color='c', linestyle=':', label='CEloss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 3]
        plt.plot(x_axis, y_axis, color='m', linestyle=':', label='sploss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 4]
        plt.plot(x_axis, y_axis, color='r', linestyle=':', label='total_loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)