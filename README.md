# AdvMask-A-Sparse-Adversarial-Attack-Based-Data-Augmentation-Method-for-Image-Classification

This is the official implementation of AdvMask (http://arxiv.org/abs/2211.16040), as was used for the paper.
 
You can directly start off using our implementations on CIFAR-10 and CIFAR-100.
## Use AdvMask for data augmentation
- Clone this directory and `cd`  into it.
 
`git clone https://github.com/Jackbrocp/AdvMask` 

`cd AdvMask`

## Updates
- 2023/7/10: Initial release

## Getting Started
### Requirements
- Python 3
- PyTorch 1.6.0
- Torchvision 0.7.0
- Numpy
<!-- Install a fitting Pytorch version for your setup with GPU support, as our implementation  -->

### Train Examples 
#### Download the Attack Masks
[CIFAR-10](https://drive.google.com/file/d/1Y7BR3--gQfeXO9S7KPe3FirbhEiAbtUk/view?usp=sharing)
[CIFAR-100](https://drive.google.com/file/d/1bqf3tMpmng-jq-JplM1hGup7_BYYmxob/view?usp=drive_link)
Download the attack mask and put them into  ```./Attack_Mask/```.
#### Parameters
```--conf```，path to the config file, e.g., ```confs/resnet18.yaml```
#### Examples 
Apply AdvMaks as data augmentation method training ResNet18 model on CIFAR10/100 dataset.
```python train.py --conf confs/resnet18.yaml```

#### More Examples
Run additional comparisons on AdvMask combined with other data augmentation methods. (e.g., "AdvMask+AutoAugment")
First change ```mask``` parameter in the config file, e.g. "AutoAugment", "Fast-AutoAugment"
```python additional_comparison.py --conf confs/resnet18.yaml```

## Citation
If you find this repository useful in your research, please cite our paper:
`@article{yang2022advmask,
  title={AdvMask: A Sparse Adversarial Attack Based Data Augmentation Method for Image Classification},
  author={Yang, Suorong and Li, Jinqiao and Zhao, Jian and Shen, Furao},
  journal={arXiv preprint arXiv:2211.16040},
  year={2022}
}`
