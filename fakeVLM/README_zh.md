# 检测假图像
基于此项目进行复现

**面向通用假图像检测器，可跨生成模型泛化** <br>
[Utkarsh Ojha*](https://utkarshojha.github.io/), [Yuheng Li*](https://yuheng-li.github.io/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/) <br>
(*同等贡献) <br>
CVPR 2023

[[项目页面](https://utkarshojha.github.io/universal-fake-detection/)] [[论文](https://arxiv.org/abs/2302.10174)]

<p align="center">
    <a href="https://utkarshojha.github.io/universal-fake-detection/">"><img src="resources/teaser.png" width="80%">></a> <br>
    使用一种生成模型的图像（例如 GAN），检测其他<i>类型</i>的假图像（例如扩散模型）
</p>

## 内容

- [安装](#setup)
- [预训练模型](#weights)
- [数据](#data)
- [评估](#evaluation)
- [训练](#training)


## 安装

1. 克隆本仓库 
```bash
git clone https://github.com/Yuheng-Li/UniversalFakeDetect
cd UniversalFakeDetect
```

2. 安装必要库
```bash
pip install torch torchvision
```

## 数据
- 你的数据集应格式化为包含(路径, 标签)对的txt文件:
  - 训练集: `/home/wangz/zgh/dataset/train_list.txt`
  - 测试集: `/home/wangz/zgh/dataset/test_list.txt`
  - 格式示例: `./dataset/real/celebahq/1323.jpg\t0` (路径和类别ID用制表符分隔)

## 评估
- 运行评估:
```bash
python validate.py  --arch=CLIP:ViT-L/14   --ckpt=pretrained_weights/fc_weights.pth   --result_folder=clip_vitl14 --real_path=/home/wangz/zgh/dataset/test_list.txt --fake_path=/home/wangz/zgh/dataset/test_list.txt
```

## 训练
- 使用11分类训练:
```bash
python train.py --name=clip_vitl14_multiclass --data_mode=your_custom_data --arch=CLIP:ViT-L/14 --n_classes=11 --fix_backbone
```
- **重要**: 使用`--n_classes=11`指定你的数据集类别数

- 我们的主要模型是在与[这项工作](https://arxiv.org/abs/1912.11035)的作者使用的相同数据集上训练的。从[这里](https://drive.google.com/file/d/1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh/view)下载官方训练数据集（约72GB）。 

- 下载并解压数据集到`datasets/train`目录。整体结构应如下所示：
```
datasets
└── train            
      └── progan            
           ├── airplane
           │── bird
           │── boat
           │      .
           │      .
```
- 总共20个不同的物体类别，每个文件夹包含相应的真伪图像在`0_real`和`1_fake`文件夹中。
- 然后可以用以下命令训练模型：
```bash
python train.py --name=clip_vitl14 --wang2020_data_path=datasets/ --data_mode=wang2020  --arch=CLIP:ViT-L/14  --fix_backbone
```
- **重要**: 不要忘记在训练时使用`--fix_backbone`参数，这确保了只有线性层的参数会被训练。

## 致谢

我们要感谢[Sheng-Yu Wang](https://github.com/PeterWang512)发布不同生成模型的真实/伪造图像。我们的训练管道也受到他的[开源代码](https://github.com/PeterWang512/CNNDetection)的启发。我们还要感谢[CompVis](https://github.com/CompVis)发布预训练的[LDMs](https://github.com/CompVis/latent-diffusion)以及[LAION](https://laion.ai/)开放源码的[LAION-400M数据集](https://laion.ai/blog/laion-400-open-dataset/)。

## 引用

如果你发现我们的工作对你的研究有帮助，请使用以下方式引用：
```bibtex
@inproceedings{ojha2023fakedetect,
      title={Towards Universal Fake Image Detectors that Generalize Across Generative Models}, 
      author={Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
      booktitle={CVPR},
      year={2023},
}

#溯源任务，在数据集