"""
An example for dataset loaders, starting with data loading including all the functions that either preprocess or postprocess data.
"""
import imageio
import torch
import torchvision.utils as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.datasets import MNIST, SVHN, CIFAR10, CIFAR100, FashionMNIST, ImageFolder

class DFKDDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
       

    def load_val_dataset(self):

        if self.config.dataset == 'MNIST':        
            self.data_test = MNIST(self.config.data,
                            train=False,
                            transform=transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]), download=True)           
            self.data_test_loader = DataLoader(self.data_test, batch_size=64, num_workers=2, shuffle=False)
        else:  
            if self.config.dataset == 'SVHN':
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4378, 0.4439, 0.4729), (0.1980, 0.2011, 0.1971)),
                ])
                self.data_test = SVHN(self.config.data,
                                'test',
                                transform=transform_test,
                                download=True)
            elif self.config.dataset == 'FMNIST':
                transform_test = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.2856,), (0.3385,)),
                ])
                self.data_test = FashionMNIST(self.config.data + '/FashionMNIST',
                                train=False,
                                transform=transform_test,
                                download=True)
            elif self.config.dataset == 'cifar10':
                transform_test = transforms.Compose([
                    transforms.Resize((32,32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                             std=[0.2471, 0.2435, 0.2616]),
                ])

                self.data_test = CIFAR10(self.config.data,
                                    train=False,
                                    transform=transform_test,
                                    download=True)

            else:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                # if self.config.dataset == 'cifar10': 
                #     self.data_test = CIFAR10(self.config.data,
                #                     train=False,
                #                     transform=transform_test,
                #                     download=True)
                if self.config.dataset == 'cifar100':
                    self.data_test = CIFAR100(self.config.data,
                                    train=False,
                                    transform=transform_test,
                                    download=True)
                if self.config.dataset == 'tiny-imagenet':
                    transform_test = transforms.Compose([
                        transforms.Resize(self.config.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
                    # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    self.data_test = ImageFolder(self.config.data + 'tiny_imagenet/val',
                                    transform=transform_test)
            self.data_test_loader = DataLoader(self.data_test,shuffle=False, batch_size=self.config.batch_size, num_workers=2)


    def load_val_dataset_wrnet(self):

        if self.config.dataset == 'MNIST':        
            self.data_test = MNIST(self.config.data,
                            train=False,
                            transform=transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]), download=True)           
            self.data_test_loader = DataLoader(self.data_test, batch_size=64, num_workers=2, shuffle=False)
        else:  
            if self.config.dataset == 'SVHN':
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4378, 0.4439, 0.4729), (0.1980, 0.2011, 0.1971)),
                ])
                self.data_test = SVHN(self.config.data,
                                'test',
                                transform=transform_test,
                                download=True)
            elif self.config.dataset == 'FMNIST':
                transform_test = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.2856,), (0.3385,)),
                ])
                self.data_test = FashionMNIST(self.config.data + '/FashionMNIST',
                                train=False,
                                transform=transform_test,
                                download=True)
            elif self.config.dataset == 'cifar10':
                transform_test = transforms.Compose([
                    transforms.Resize((32,32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.2023, 0.1994, 0.2010]),
                ])

                self.data_test = CIFAR10(self.config.data,
                                    train=False,
                                    transform=transform_test,
                                    download=True)

            else:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])

                # if self.config.dataset == 'cifar10': 
                #     self.data_test = CIFAR10(self.config.data,
                #                     train=False,
                #                     transform=transform_test,
                #                     download=True)
                if self.config.dataset == 'cifar100':
                    self.data_test = CIFAR100(self.config.data,
                                    train=False,
                                    transform=transform_test,
                                    download=True)
                # if self.config.dataset == 'tiny-imagenet':
                #     transform_test = transforms.Compose([
                #         transforms.Resize(self.config.img_size),
                #         transforms.ToTensor(),
                #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                #     ])
                #     # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                #     self.data_test = ImageFolder(self.config.data + 'tiny_imagenet/val',
                #                     transform=transform_test)
            self.data_test_loader = DataLoader(self.data_test,shuffle=False, batch_size=self.config.batch_size, num_workers=2)

    def load_train_dataset(self):
        if self.config.dataset == 'MNIST':        
            self.data_test = MNIST(self.config.data,
                            train=True,
                            transform=transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]), download=True)           
            self.data_test_loader = DataLoader(self.data_test, batch_size=64, num_workers=2, shuffle=False)
        else:  
            if self.config.dataset == 'SVHN':
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4378, 0.4439, 0.4729), (0.1980, 0.2011, 0.1971)),
                ])
                self.data_test = SVHN(self.config.data,
                                'test',
                                transform=transform_test,
                                download=True)
            elif self.config.dataset == 'FMNIST':
                transform_test = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.2856,), (0.3385,)),
                ])
                self.data_test = FashionMNIST(self.config.data + '/FashionMNIST',
                                train=True,
                                transform=transform_test,
                                download=True)
            else:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                if self.config.dataset == 'cifar10': 
                    self.data_test = CIFAR10(self.config.data,
                                    train=True,
                                    transform=transform_test,
                                    download=True)
                if self.config.dataset == 'cifar100':
                    self.data_test = CIFAR100(self.config.data,
                                    train=True,
                                    transform=transform_test,
                                    download=True)
                if self.config.dataset == 'tiny-imagenet':
                    transform_test = transforms.Compose([
                        transforms.Resize(self.config.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
                    self.data_test = ImageFolder(self.config.data + 'tiny_imagenet/train',
                                    transform=transform_test)
            self.data_test_loader = DataLoader(self.data_test,shuffle=True, batch_size=self.config.batch_size, num_workers=2)





    # def plot_samples_per_epoch(self, batch, epoch):
    #     """
    #     Plotting the batch images
    #     :param batch: Tensor of shape (B,C,H,W)
    #     :param epoch: the number of current epoch
    #     :return: img_epoch: which will contain the image of this epoch
    #     """
    #     img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
    #     v_utils.save_image(batch,
    #                        img_epoch,
    #                        nrow=4,
    #                        padding=2,
    #                        normalize=True)
    #     return imageio.imread(img_epoch)

    # def make_gif(self, epochs):
    #     """
    #     Make a gif from a multiple images of epochs
    #     :param epochs: num_epochs till now
    #     :return:
    #     """
    #     gen_image_plots = []
    #     for epoch in range(epochs + 1):
    #         img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
    #         try:
    #             gen_image_plots.append(imageio.imread(img_epoch))
    #         except OSError as e:
    #             pass

    #     imageio.mimsave(self.config.out_dir + 'animation_epochs_{:d}.gif'.format(epochs), gen_image_plots, fps=2)

    def finalize(self):
        pass

