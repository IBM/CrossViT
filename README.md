# CrossViT

This repository is the official implementation of CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification. [ArXiv](https://arxiv.org/abs/2103.14899)

If you use the codes and models from this repo, please cite our work. Thanks!

@inproceedings{
    chen2021crossvit,
    title={{CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification}},
    author={Chun-Fu (Richard) Chen and Rameswar Panda and Quanfu Fan},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2021}
}
 

## Image Classification

### Installation

To install requirements:

```setup
pip install -r requirements.txt
```

### Data preparation

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

### Model Zoo

We provide models trained on ImageNet1K.

| Name | Acc@1 (%) | FLOPs (G)  | Params (M) | Url |
| --- | --- | --- | --- | --- |
| CrossViT-9$\dagger$ | 77.1 | 2.0 | 8.8M | [model]() |
| CrossViT-15$\dagger$ | 82.3 | 6.1 | 28.2M | [model]() |
| CrossViT-18$\dagger$ | 82.8 | 9.5 | 44.3M | [model]() |


### Training

To train CrossViT-9$\dagger$ on ImageNet on a single node with 8 gpus for 300 epochs run:

```shell script

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model crossvit_9_dagger_224 --batch-size 256 --data-path /path/to/imagenet
```

Other model names can be found at [models/crossvit.py](models/crossvit.py).

### Multinode training

Distributed training is available via Slurm and `submitit`:

To train a CrossViT-9$\dagger$ model on ImageNet on 4 nodes with 8 gpus each for 300 epochs:

```
python run_with_submitit.py --model crossvit_9_dagger_224 --data-path /path/to/imagenet --batch-size 128 --warmup-epochs 30
```

Note that: some slurm configurations might need to be changed based on your cluster.


### Evaluation

To evaluate a pretrained model on `crossvit_9_dagger_224`:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model crossvit_9_dagger_224 --batch-size 128 --data-path /path/to/imagenet --eval --pretrained
```