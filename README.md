<h1 align="center"> SARATR-X: Towards Building A Foundation Model for SAR Target Recognition </h1> 

<h5 align="center"><em> Weijie Li (李玮杰), Wei Yang (杨威), Yuenan Hou (侯跃南), Li Liu (刘丽), Yongxiang Liu (刘永祥), and Xiang Li (黎湘) </em></h5>

<p align="center">
  <a href="#Introduction">Introduction</a> |
  <a href="#Pre-training">Pre-training</a> |
  <a href="#Classification">Classification</a> |
  <a href="#Detection">Detection</a> |
  <a href="#Statement">Statement</a>
</p >
<p align="center">
<a href="https://ieeexplore.ieee.org/document/10856784"><img src="https://img.shields.io/badge/Paper-IEEE%20TIP-blue"></a>
<a href="https://arxiv.org/abs/2405.09365"><img src="https://img.shields.io/badge/Paper-arxiv-red"></a>  
<a href="https://pan.baidu.com/s/1EuTo0WCbb69F_bGV6Iwd8A"><img src="https://img.shields.io/badge/Data&Checkpoint-BaiduYun-blue"></a>
<a href="https://huggingface.co/waterdisappear/SARATR-X"><img src="https://img.shields.io/badge/Data&Checkpoint-Hugging%20face-yellow"></a>
</p>


## Introduction

This is the official repository for the paper “SARATR-X: Towards Building A Foundation Model for SAR Target Recognition”.  If you find our work is useful, please give us a star 🌟 in GitHub and cite our paper in the BibTex format at the end.

这里是论文 “SARATR-X: Towards Building A Foundation Model for SAR Target Recognition (SARATR-X：迈向SAR目标识别基础模型) ”的代码库。如果您觉得我们的工作有价值，请在 GitHub 上给我们个星星 🌟 并按页面最后的 BibTex 格式引用我们的论文。

You can contact us to get these datasets and weights by email: lwj2150508321@sina.com.

可以通过给我们发送邮件lwj2150508321@sina.com获取相关数据集和权重。

<figure>
<div align="center">
<img src=example/fig_radarmap.png width="70%">
</div>
</figure>

**Abstract:** 
Despite the remarkable progress in synthetic aperture radar automatic target recognition (SAR ATR), recent efforts have concentrated on detecting and classifying a specific category, e.g., vehicles, ships, airplanes, or buildings. One of the fundamental limitations of the top-performing SAR ATR methods is that the learning paradigm is supervised, task-specific, limited-category, closed-world learning, which depends on massive amounts of accurately annotated samples that are expensively labeled by expert SAR analysts and have limited generalization capability and scalability. In this work, we make the first attempt towards building a foundation model for SAR ATR, termed SARATR-X. SARATR-X learns generalizable representations via self-supervised learning (SSL) and provides a cornerstone for label-efficient model adaptation to generic SAR target detection and classification tasks. Specifically, SARATR-X is trained on 0.18 M unlabelled SAR target samples, which are curated by combining contemporary benchmarks and constitute the largest publicly available dataset till now. Considering the characteristics of SAR images, a backbone tailored for SAR ATR is carefully designed, and a two-step SSL method endowed with multi-scale gradient features was applied to ensure the feature diversity and model scalability of SARATR-X. The capabilities of SARATR-X are evaluated on classification under few-shot and robustness settings and detection across various categories and scenes, and impressive performance is achieved, often competitive with or even superior to prior fully supervised, semi-supervised, or self-supervised algorithms. 

**摘要：** 
尽管合成孔径雷达自动目标识别（synthetic aperture radar automatic target recognition, SAR ATR）取得了显著进展，但最近的工作主要集中在对特定类别（如车辆、船舶、飞机或建筑物）的检测和分类上。性能良好的 SAR ATR 方法的一个基本局限是，其学习范式是有监督的、特定任务的、有限类别的、封闭世界的学习，这种学习依赖于大量准确标注的样本，而这些样本是由 SAR 专家分析人员花费高昂成本标注的，其泛化能力和可扩展性有限。在这项工作中，我们首次尝试为 SAR ATR 建立一个基础模型，称为 SARATR-X。SARATR-X 通过自监督学习 (self-supervised learning, SSL) 学习可泛化的表征，为标签高效模型适应通用 SAR 目标检测和分类任务提供了基石。具体来说，SARATR-X 在 0.18 M 个未标记的合成孔径雷达目标样本上进行预训练，这些样本是结合当代数据集基准，构成了迄今为止最大的公开可用预训练数据集。考虑到合成孔径雷达图像的特点，为合成孔径雷达 ATR 量身定制的骨架经过了精心设计，并采用了具有多尺度梯度特征的两步 SSL 方法，以确保 SARATR-X 的特征多样性和模型可扩展性。我们对 SARATR-X 的能力进行了评估，包括少样本和鲁棒性设置下的分类以及各种类别和场景的检测，其性能令人印象深刻，通常可与之前的全监督、半监督或自监督算法相媲美，甚至更胜一筹。

## Pre-training
Our codes are based on [SAR-JEPA](https://github.com/waterdisappear/SAR-JEPA) and [HiVit](https://github.com/zhangxiaosong18/hivit). 

### Requirements:
- Python3
- CUDA 11.1
- PyTorch 1.8+ with CUDA support
- timm 0.5.4
- tensorboard

### Step-by-step installation (new version for h800 mm3.0 not suit for current Github version in paper, please see my new saratrx_mmdet3.0.7z file in baiduyun and huggfaceing)

```bash
# 安装预训练环境
# pretrain 
conda create -n saratrx python=3.10 -y
conda activate saratrx
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.5.4 tensorboard
pip install opencv-python opencv-python-headless
pip install numpy==1.26.4

# 安装检测环境
# detection 
cd detection
pip install "setuptools<80.9"
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install -v -e .
pip install pycocotools==2.0.7

# 安装分类环境
# classifiction 
cd classification
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
```

### Step-by-step installation (old version for 3090)

```bash
conda create -n saratrx python=3.9 -y
conda activate saratrx
cd pre-training

pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install timm==0.5.4 tensorboard
pip install -r requirements_pretrain.txt
```

### Pre-training dataset

|                         **Dataset**                          | **Year** | **Task** | **#Imgs.** | **Img. Size** | **#Targets** | **#Scenes** | **Res. (m)** | **Band** | **Pol.** |              **Target description**               |
| :----------------------------------------------------------: | :------: | :------: | :--------: | :-----------: | :----------: | :---------: | :----------: | :------: | :------: | :-----------------------------------------------: |
| [MSTAR](https://www.sdms.afrl.af.mil/index.php?collection=mstar) |   1995   |  Cls\.   |   14,577   |    128~193    |      10      |      1      |     0\.3     |    X     |  Single  |           Fine\-grained vehicle dataset           |
| [Sandia MiniSAR](https://www.sandia.gov/radar/complex-data/index.html) |   2006   |  Det\.   |   3,927    |      224      |      ≥1      |     ≥7      |     0\.1     |    Ku    |  Single  | Terrestrial targets in urban, deserts, and others |
| [SARSim](https://ieeexplore.ieee.org/abstract/document/7968358) |   2017   |  Cls\.   |   21,168   |      139      |      14      |      3      |     0\.3     |    X     |  Single  |            Simulation vehicle dataset             |
| [SAMPLE](https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public) |   2019   |  Cls\.   |   5,380    |      128      |      10      |      2      |     0\.3     |    X     |  Single  |      Simulation and measured vehicle dataset      |
|        [SIVED](https://github.com/CAESAR-Radi/SIVED)         |   2023   |  Det\.   |   1,044    |      512      |      ≥1      |     ≥4      |  0\.1~0\.3   | X/Ku/Ka  |  Single  |             Synthetic vehicle dataset             |
|         [OpenSARShip](https://opensar.sjtu.edu.cn/)          |   2017   |  Cls\.   |   26,679   |     9~445     |      14      |     10      |  2\.3~17\.4  |    C     |  Double  |             Fine\-grained ship slices             |
| [SAR\-Ship](https://github.com/CAESAR-Radi/SAR-Ship-Dataset) |   2019   |  Det\.   |   39,729   |      256      |      ≥1      |     ≥4      |     3~25     |    C     |   Quad   |          Ship dataset in complex scenes           |
| [AIR\-SARShip](https://radars.ac.cn/en/article/doi/10.12000/JR19097) |   2019   |  Det\.   |    801     |   512~1000    |      ≥1      |     ≥3      |     1~3      |    C     |  Single  |          Ship dataset in complex scenes           |
|       [HRSID](https://github.com/chaozhong2010/HRSID)        |   2020   |  Det\.   |   5,604    |      800      |      ≥1      |     ≥2      |    0\.5~3    |   C/X    |   Quad   |           Instance\-level ship dataset            |
|  [SSDD](https://github.com/TianwenZhang0825/Official-SSDD)   |   2021   |  Det\.   |   1,160    |    214~668    |      ≥1      |     ≥2      |     1~15     |   C/X    |   Quad   |                   Ship dataset                    |
|   [SADD](https://github.com/hust-rslab/SAR-aircraft-data)    |   2022   |  Det\.   |    883     |      224      |      ≥1      |     ≥2      |    0\.5~3    |    X     |  Single  |                 Aircraft dataset                  |
| [SAR\-AIRcraft](https://radars.ac.cn/cn/article/doi/10.12000/JR23043) |   2023   |  Det\.   |   18,818   |      512      |      ≥7      |     ≥3      |      1       |    C     |  Single  |                  Aircraft ataset                  |
| [MSAR](https://radars.ac.cn/web/data/getData?dataType=MSAR)  |   2022   |  Det\.   |   28,499   |   256~2048    |      ≥4      |     ≥6      |      1       |    C     |   Quad   |         Terrestrial and maritime targets          |
| [OGSOD](https://github.com/mmic-lcl/Datasets-and-benchmark-code) |   2023   |  Det\.   |   18,331   |      256      |      ≥3      |     ≥2      |      3       |    C     |  Double  | Targets include bridges, oil tanks, and harbours  |

### Start Pre-training with SAR images

We consider that constructing self-supervised signals for SAR images, which suffer from many disturbances in imaging quality, requires a combination of noise suppression and feature compression. And the initial weight is from [mae_hivit_base_1600ep.pth](https://github.com/zhangxiaosong18/hivit).

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 --use_env main_pretrain.py
    --data_path <imagenet-path> --output_dir <pertraining-output-path>
    --model mae_hivit_base_dec512d6b --mask_ratio 0.75
    --batch_size 100 --accum_iter 1 --blr 1.5e-4 --weight_decay 0.05 --epochs 800 --warmup_epochs 5 
```
Q1: How do I use my dataset?

A1: Please change the --data_path and modify the data load code if needed in [main_pretrain.py](https://github.com/waterdisappear/SARATR-X/blob/main/pre-training/main_pretrain.py) and [datasets.py](https://github.com/waterdisappear/SARATR-X/blob/main/pre-training/util/datasets.py).

```python
    # Dataset parameters
    parser.add_argument('--data_path', default='D:\\2023_SARatrX_1\Pre-Train Data\\186K_notest\\', type=str,
                        help='dataset path')
                        
    from util.datasets import load_data
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path), transform=transform_train)
    dataset_train = load_data(os.path.join(args.data_path), transform=transform_train)
    print(len(dataset_train))
```

Q2: How do we make improvements?

A2: You can add more high-quality data and try more data augment methods. Besides, we suggest improvements to the HiViT's attention mechanism in [models_hivit.py](https://github.com/waterdisappear/SARATR-X/blob/main/pre-training/models/models_hivit.py) and our proposed SAR target features in [models_hivit_mae.py](https://github.com/waterdisappear/SARATR-X/blob/main/pre-training/models/models_hivit_mae.py).

```python
    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(contrast=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    ])
    
    # SAR feature 
    self.sarfeature1 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=9,
                              img_size=self.img_size,patch_size=self.patch_size)
    self.sarfeature2 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=13,
                              img_size=self.img_size,patch_size=self.patch_size)
    self.sarfeature3 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=17,
                              img_size=self.img_size,patch_size=self.patch_size)
    target = torch.cat([self.patchify(self.sarfeature1(imgs)), self.patchify(self.sarfeature2(imgs)), self.patchify(self.sarfeature3(imgs))], dim=-1)
```
Q3: How to load ImageNet pre-training weights?

A3: You can see in [main_pretrain.py](https://github.com/waterdisappear/SARATR-X/blob/main/pre-training/main_pretrain.py).

```python
    # define the model
    model = models.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    checkpoint = torch.load('./mae_hivit_base_1600ep.pth',
                            map_location='cpu')
    # load pre-trained model
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
```

## Classification

Our few-shot learning is based on [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch). You need to install this and use our modified ''Dassl.pytorch\dassl\utils\tools.py'' and ''Dassl.pytorch\dassl\data\transforms\transforms.py'' in our modified [zip](https://github.com/waterdisappear/SARATR-X/blob/main/classification/Dassl.pytorch.zip) for SAR single-channel amplitude images. Then, you can run our [MIM_linear.sh](https://github.com/waterdisappear/SARATR-X/blob/main/classification/finetune/MIM_linear.sh) for evaluations.

### Linear probing with SAR-VSA

Three open-source target datasets (MSTAR, FUSARSship, and SAR-ACD) were utilized by first constructing a fine-grained classification dataset, SAR-VSA, with 25 categories to evaluate the effectiveness of the proposed improvements. 

| **Fine\-grained category**  | **#Train** | **#Test** |
| :-------------------------: | :--------: | :-------: |
|  anti\-aircraft \(ZSU234\)  |    299     |    274    |
|      bulldozer \(D7\)       |    299     |    274    |
|      howitzer \(2S1\)       |    299     |    274    |
|  infantry vehicle \(BMP2\)  |    698     |    587    |
|  main battle tank \(T62\)   |    299     |    273    |
|  main battle tank \(T72\)   |    691     |    582    |
|    patrol car \(BRDM2\)     |    298     |    274    |
| personnel carrier \(BTR60\) |    256     |    195    |
| personnel carrier \(BTR70\) |    233     |    196    |
|      truck \(ZIL131\)       |    299     |    274    |
|           bridge            |   1,023    |    438    |
|        coastal land         |    707     |    303    |
|         land patch          |   1,137    |    487    |
|      sea clutter wave       |   1,378    |    590    |
|          sea patch          |   1,250    |    535    |
|       ship \(cargo\)        |    366     |    156    |
|      ship \(fishing\)       |    248     |    106    |
|       ship \(tanker\)       |    150     |    64     |
|       ship \(others\)       |    312     |    133    |
|     strong false alarms     |    299     |    128    |
|  aircraft \(Airbus A220\)   |     91     |    373    |
|  aircraft \(Airbus A330\)   |     97     |    415    |
|  aircraft \(Comac ARJ21\)   |    103     |    411    |
|   aircraft \(Boeing 737\)   |    100     |    428    |
|   aircraft \(Boeing 787\)   |    113     |    391    |

### MSTAR‘s SOC and EOC

**Modification on Dassl.** We perform center cropping for different slice sizes of the MSTAR dataset to prevent correlation between target categories and image sizes during resizing.

    # Dassl.pytorch.zip\Dassl.pytorch\dassl\data\transforms\transforms.py
    
    def _build_transform_test(cfg, choices, target_size, normalize):
        if cfg.OUTPUT_DIR.split('/')[-4].split('_')[0] == 'MSTAR':
            tfm_test += [CenterCrop([128, 128])]


## Detection

We use MMDetection.

### Requirements:

- Python3
- CUDA 11.1
- PyTorch 1.8+ with CUDA support
- timm 0.5.4
- mmcv-full 1.6.0
- opencv-python
- apex

### Step-by-step installation (old version for 3090)

```bash
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8/index.html
pip install opencv-python timm==0.5.4
sh ../install_apex.sh

pip install -e .
pip install -r requirements_detection.txt
```

## Fine-tuning
```bash
chmod -R +x tools
./tools/dist_train.sh configs/_hivit_/hivit_base_SARDet.py 8 --work-dir ./work_dirs/SARDet
```

## Statement


- This project is released under the [CC BY-NC 4.0](LICENSE).
- 本项目采用[CC BY-NC 4.0](LICENSE)协议发布。
- If you have any questions or need additional data, code and weight files, please contact us at lwj2150508321@sina.com. 
- 如有任何问题或者需要其他数据、代码和权重文件，请通过 lwj2150508321@sina.com 联系我们。
- If you find our work is useful, please give us 🌟 in GitHub and cite our paper in the following BibTex format:
- 如果您觉得我们的工作有价值，请在 GitHub 上给我们 🌟 并按以下 BibTex 格式引用我们的论文：
```
@ARTICLE{li2025saratr,
  author={Li, Weijie and Yang, Wei and Hou, Yuenan and Liu, Li and Liu, Yongxiang and Li, Xiang},
  journal={IEEE Transactions on Image Processing}, 
  title={SARATR-X: Toward Building a Foundation Model for SAR Target Recognition}, 
  year={2025},
  volume={34},
  number={},
  pages={869-884},
  doi={10.1109/TIP.2025.3531988}}

@ARTICLE{li2024predicting,
  title = {Predicting gradient is better: Exploring self-supervised learning for SAR ATR with a joint-embedding predictive architecture},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {218},
  pages = {326-338},
  year = {2024},
  issn = {0924-2716},
  doi = {https://doi.org/10.1016/j.isprsjprs.2024.09.013},
  url = {https://www.sciencedirect.com/science/article/pii/S0924271624003514},
  author = {Li, Weijie and Yang, Wei and Liu, Tianpeng and Hou, Yuenan and Li, Yuxuan and Liu, Zhen and Liu, Yongxiang and Liu, Li}}
```
