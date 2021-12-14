# Audio Captioning Transformer
This repository contains source code for our paper [Audio Captioning Transformer](https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Mei_68.pdf).
## Set up environment
* Create a conda environment with dependencies: `conda env create -f environment.yml -n name`
* All of our experiments are running on RTX 30 series GPUs with CUDA11. This environment may just work for RTX 30x GPUs.
## Set up dataset
All the experiments were carried out on AudioCaps dataset, which is sourced from AudioSet.
Our download version contains 49274/49837 audio clips in training set, 494/495 audio clips in validation set, 957/975 audio clips in test set.

For reproducibility, our downloaded version can be accessed at: https://pan.baidu.com/s/1DkGsfQ0aM6lx6Gf6gCyrVw  password: a1p4 

To prepare the dataset:
* Put downloaded zip files under `data` directory, and run `data_unzip.sh` to extract the zip files.
* Run `python data_prep.py` to create h5py files of the dataset.

## Prepare evaluation tool

* Run `coco_caption/get_stanford_models.sh` to download the libraries necessary for evaluating the metrics.

## Experiments 

### Training

* The default setting is for 'ACT_m_scratch'
* Run experiments: `python train.py -n exp_name`
* Set the parameters you want in `settings/settings.yaml`

#### Pretrained encoder

We provide two pretrained encoders, one is a pretrained DeiT model, another is the DeiT model pretrained on AudioSet.
1. [DeiT model](https://drive.google.com/file/d/1eA3SYO2n9soU5AB6YxMNGDjno5E5AN7Q/view?usp=sharing)
2. [DeiT model pretrained on AudioSet](https://drive.google.com/file/d/1QgQLbeBHwly5UN_V15mSJZ812h6QIgFe/view?usp=sharing)

To use pretrained encoder:
* Download the pretrained encoder models and put them under the directory `pretrained_models`
* Set settings in `settings/settings,yaml`
  * set `encoder.model:` to 'deit' or 'audioset'
  * set `encoder.pretrained` to 'Yes'
  * set `path.encoder` to the model path, e.g. 'pretrained_models/deit.pth'
* Run experiments

### Reproducible results

As we have refactored the code and made some improvements after the DECASE workshop, there are little differences among the reproducible results and those reported in the paper (the metrics are higher now), the conclusions are the same.

We provide three pretrained models, those are all trained using a pre-trained encoder.
1. [ACT_s(SPIDEr:0.4244)](https://drive.google.com/file/d/1c1b1Q6hAprt-CCu4VtMSb4Jqk604-RHi/view?usp=sharing)
2. [ACT_m(SPIDEr:0.4178)](https://drive.google.com/file/d/1nYym4APxEX4aiHINEykUIPyoHvXIjJRF/view?usp=sharing)
3. [ACT_l(SPIDEr:0.4257)](https://drive.google.com/file/d/1fF4_XnheFiz_tPaRVdz4q5eUUi8infg_/view?usp=sharing)

To get the reproducible results:
* Download the pretrained models and put them under the directory `pretrained_models`
* Set settings in `settings/settings,yaml`
  * set `mode` to 'eval'
  * set `path.eval_model` to the model path
* Run experiments


## Cite
If you wish to cite this work, please kindly cite the following paper:
```
@inproceedings{Mei2021act,
    author = "Mei, Xinhao and Liu, Xubo and Huang, Qiushi and Plumbley, Mark D. and Wang, Wenwu",
    title = "Audio Captioning Transformer",
    booktitle = "Proceedings of the 6th Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021)",
    address = "Barcelona, Spain",
    month = "November",
    year = "2021",
    pages = "211--215",
    isbn = "978-84-09-36072-7",
    doi. = "10.5281/zenodo.5770113"
}
```