import os

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from matplotlib import pyplot as plt


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的⽂本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图⽚张量
            ax.imshow(img.numpy())
        else:
            # PIL图⽚
            ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i])
    plt.show()
    return axes

trans = transforms.ToTensor()
os.makedirs('./data', exist_ok=True)
mnist_train = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    transform=trans,
    download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    transform=trans,
    download=True)

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
