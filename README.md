# Transformer-based Representation for Face Attribute 

The zip file contains source codes we use in our paper for testing the facial attribute evaluation accuracy on CelebA dataset.

## Dependencies

* Anaconda3 (Python3.6, with Numpy etc.)
* Pytorch1.10.0
* tensorboard, tensorboardX

More details about dependencies are shown in requirements.txt.

## Datasets

[Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a large-scale face attributes dataset with more than **200K** celebrity images, each with **40** attribute annotations. The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including

- **10,177** number of **identities**,
- **202,599** number of **face images**, and
- **5 landmark locations**, **40 binary attributes** annotations per image.

The dataset can be employed as the training and test sets for the following computer vision tasks: face attribute recognition, face recognition, face detection, landmark (or facial part) localization, and face editing & synthesis

## Features

* Swin Transformer is used as the backbone for shared attribute feature learning.
* All the 40 facial attributes are reordered according to the proposed attributes grouping strategy.

## Usage

### Download pretrained model

| Model   | Download                                                     |
| ------- | ------------------------------------------------------------ |
| TransFA | [MEGA](https://mega.nz/file/EbElxLrJ#NfuUejK2nrOBWrbnVujdWF4QWNHdVl-XPGVshzOVnI4) |

After downloading the pretrained model, we should put the model to `./pretrained`

### Download dataset

| Dataset Name | Download                                                   | Images  |
| ------------ | ---------------------------------------------------------- | ------- |
| CelebA       | [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | 202,599 |

After downloading the whole dataset, you can unzip **img_align_celeba.zip** to the `./images`.

### Test the model

```
./run_test.sh
```
