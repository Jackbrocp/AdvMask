""" helper function

author baiyu
"""
import os
import sys
import pickle
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import organize_transform
sys.path.append('../')    
from adv_mask import *
def load_CIFAR_100(root, train=True, fine_label=True):
    if train:
        filename = root + 'my_train'
    else:
        filename = root + 'test'
 
    with open(filename, 'rb')as f:
        datadict = pickle.load(f,encoding='bytes')
 
        if train:
            # [50000, 32, 32, 3]
            X = datadict['data']
            filename_list = datadict['filenames']
            X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1)
            Y = datadict['labels']
            Y = np.array(Y)
            return X, Y, filename_list
        else:
            # [10000, 32, 32, 3]
            X = datadict[b'data']
            filename_list = datadict[b'filenames']
            X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1)
            Y = datadict[b'fine_labels']
            Y = np.array(Y)
            return X, Y
 
 
class CIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, fine_label=True, transform=True):
        if train:
            self.data,self.labels,self.filename_list=load_CIFAR_100(root,train,fine_label=fine_label)
        else:
            self.data,self.labels = load_CIFAR_100(root,train,fine_label=fine_label)
        self.transform = transform
        self.train = train
    def __getitem__(self, index):
        if self.train:
            img, target, filename = self.data[index], int(self.labels[index]),self.filename_list[index]
            if self.transform is not None:
                img = self.transform(img)
            return img, target, filename
        else:
            img, target = self.data[index], int(self.labels[index])
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        return len(self.data)
from Network import *
def get_model(net_name, num_classes=100, local_rank=-1):
    if net_name == 'resnet18':
        model = ResNet18()
    elif net_name == 'resnet50':
        model = ResNet50()
    elif net_name == 'resnet101':
        model = ResNet101()
    elif net_name == 'resnet32':
        model = ResNet32()
    elif net_name == 'resnet44':
        model = ResNet44()
    elif net_name == 'resnet56':
        model = ResNet56()
    elif net_name == 'resnet110':
        model = ResNet110()
    elif net_name =='wresnet28_10':
        model = Wide_ResNet( 28, 10, 0.3, num_classes)
    elif net_name == 'wresnet40_2':
        model = Wide_ResNet( 40, 2, 0.3, num_classes)
    elif net_name == 'shakeshake26_2x32d':
        model = ShakeResNet(26,32,num_classes) 
    elif net_name == 'shakeshake26_2x64d':
        model = ShakeResNet(26,64,num_classes)
    elif net_name == 'shakeshake26_2x96d':
        model =  ShakeResNet(26, 96, num_classes)
    elif net_name =='shakeshake26_2x112d':
        model = ShakeResNet(26,112,num_classes)
    return model

def get_training_dataloader( batch_size=16, num_workers=2, shuffle=True, mask=''):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    ])
    if mask == 'cutout':
        transform_train.transforms.append(Cutout(1,8))
    elif mask == 'gridmask':
        transform_train.transforms.append(gridmask.Grid(d1=24,d2=33,rotate=1,ratio=0.4,mode=1,prob=1.))
    elif mask == 'has':
        transform_train.transforms.append(HaS())
    elif mask == 'randomerasing':
        transform_train.transforms.append(RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.3, ))
    elif mask == 'autoaugment':
        transform_train,_ = organize_transform.make_transform('cifar100',mask)
    elif mask== 'fast-autoaugment':
        transform_train, _ = organize_transform.make_transform('cifar100',mask)
    print(transform_train)
    root = 'data/cifar-100-python/'
    trainset = CIFAR100Dataset(root, train=True,fine_label=True,transform=transform_train)
    cifar100_training_loader = DataLoader(
        trainset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    return cifar100_training_loader

def get_test_dataloader( batch_size=16, num_workers=2, shuffle=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    ])
    root = 'data/cifar-100-python/'
    testset = CIFAR100Dataset(root,train=False,fine_label=True,transform=transform_test)
    cifar100_test_loader = DataLoader(
        testset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def get_cifar10_dataloader(batch_size=16, num_workers=2, shuffle=False,mask=''):
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    ])
    if mask == 'cutout':
        transform_train.transforms.append(Cutout(1,8))
    elif mask == 'has':
        transform_train.transforms.append(HaS())
    elif mask == 'gridmask':
        transform_train.transforms.append(gridmask.Grid(d1=24,d2=33,rotate=1,ratio=0.4,mode=1,prob=1.))
    elif mask == 'randomerasing':
        transform_train.transforms.append(RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.3, ))
    elif mask == 'autoaugment':
        transform_train,_ = organize_transform.make_transform('cifar100',mask)
    elif mask== 'fast-autoaugment':
        transform_train, _ = organize_transform.make_transform('cifar100',mask)
    root = 'data/'
    trainset = Dataset_online(root=root,
                    train=True,
                    transform=transform_train)
    testset = Dataset_online(root=root,
                        train=False,
                        transform=transform_test)
    train_loader=DataLoader(dataset=trainset, batch_size=batch_size,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True)
    test_loader=DataLoader(dataset=testset, batch_size=batch_size,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True)
    return train_loader, test_loader
class Dataset_online(Dataset):
    base_folder='cifar-10-batches-py'
    preix='adv_'
    train_list=[
        [preix+'data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        [preix+'data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        [preix+'data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        [preix+'data_batch_4', '634d18415352ddfa80567beed471001a'],
        [preix+'data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list=[
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    def __init__(self,root,train=True,transform=None, target_transform=None):
        super(Dataset_online,self).__init__()
        self.train=train
        self.root=root
        self.data: Any = []
        self.targets=[]
        self.filename_list=[]

        self.transform=transform
        self.target_transform=target_transform
        if self.train:
            data_list = self.train_list
        else:
            data_list = self.test_list
        
        for file_name, checksum in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f,encoding='latin1')
                self.data.append(entry['data'])
                if self.train:
                    self.filename_list.extend(entry['filenames'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  
        self._load_meta()
    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        """if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')"""
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
    
    def __getitem__(self, index:int):
        if self.train:
            img, target, filename = self.data[index], int(self.targets[index]), self.filename_list[index]
        else:
            img, target = self.data[index], int(self.targets[index])
        # img =  Image.fromarray((img*255).astype(np.uint8))
        if self.transform is not None :
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.train:
            return img, target, filename
        else:
            return img, target
    def __len__(self):
        return len(self.data)
    
    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
 


if __name__=='__main__':
    root = 'data/cifar-100-python/'
    dataset = CIFAR100Dataset(root, train=True,fine_label=True,transform=transforms.ToTensor())
