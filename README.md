# CrossViT: 

Before using it, make sure you have the pytorch-image-models package [`timm==0.4.9`](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman) installed. Note that our work relies of the augmentations proposed in this library. In particular, the RandAugment and RandErasing augmentations that we invoke are the improved versions from the timm library, which already led the timm authors to report up to 79.35% top-1 accuracy with Imagenet training for their best model, i.e., an improvement of about +1.5% compared to prior art. 

# Usage

Then, install PyTorch 1.7.1+ and torchvision 0.8.2+ and [pytorch-image-models 0.4.9](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Training
To train CrossViT-small on ImageNet on a single node with 8 gpus for 300 epochs run:

CrossViT-small
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model crossvit_small_224 --batch-size 256 --data-path /path/to/imagenet
```

# License
This repository is released under the CC-BY-NC 4.0. license as found in the [LICENSE](LICENSE) file.
