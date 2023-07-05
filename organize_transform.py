import sys
import torchvision.transforms as transforms
 
sys.path.append('../')   
from adv_mask import CIFAR10Policy, ImageNetPolicy
from adv_mask import fast_autoaugment
def make_transform(dataset, aug):
    if dataset=='cifar10' or dataset=='cifar100':
        if aug == 'autoaugment':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
        elif aug == 'fast-autoaugment':
            transform, transform_test = fast_autoaugment.get_transform(dataset)
    elif dataset=='tiny-imagenet':
        if aug == 'autoaugment':
            transform=transforms.Compose([
                 transforms.RandomResizedCrop(224), 
                 transforms.RandomHorizontalFlip(), 
                 ImageNetPolicy(), 
                 transforms.ToTensor(), 
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif aug == 'fast-autoaugment':
            transform, transform_test = fast_autoaugment.get_transform(dataset)
    return transform, transform_test
            
            