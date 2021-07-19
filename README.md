# TCIC

This is the implementation for *TCIC: Theme Concepts Learning Cross Language and Vision for Image Captioning* in IJCAI2021.

## Requirements and Installation
Our implementation follows [Fairseq-v0.9.0](https://github.com/pytorch/fairseq) and [fairseq-image-captioning](https://github.com/krasserm/fairseq-image-captioning).

```
sudo apt-get install -y openjdk-8-jdk
pip install --user fairseq==0.9.0
pip install --user pandas h5py scikit-learn matplotlib scikit-image sacremoses subword-nmt
export PYTHONIOENCODING=utf-8
```

You can refer to the [fairseq-image-captioning](https://github.com/krasserm/fairseq-image-captioning) for data downloading and preprocessing.


We store the directory of *cider*, *coco-caption* and *data-bin* in
[BaiduDisk](https://pan.baidu.com/s/1zMHZN7V55LsSMiLp4HLubQ), code is 3eje.
You should download them to *TCIC*.


## Training

You need run the following code in the directory of *TCIC*.

```
# Training with Cross-Entropy
bash src/scripts/train/train.sh

# Train with Reinforcement Learning
bash src/scripts/train/train.sh
```



