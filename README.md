<div align="center">

# Symphonies (Scene-from-Insts) 🎻

### [Symphonize 3D Semantic Scene Completion with Contextual Instance Queries](https://arxiv.org/abs/2306.15670)

[Haoyi Jiang](https://github.com/npurson) <sup>1,✢</sup>,
[Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ) <sup>1,✢</sup>,
Naiyu Gao <sup>2</sup>,
Haoyang Zhang <sup>2</sup>,
[Tianwei Lin](https://wzmsltw.github.io/) <sup>2</sup>,
[Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/) <sup>1</sup>,
[Xinggang Wang](https://xwcv.github.io/) <sup>1,✉️</sup>
<br>
<sup>1</sup> [School of EIC, HUST](http://english.eic.hust.edu.cn/),
<sup>2</sup> [Horizon Robotics](https://en.horizonrobotics.com/)

[**CVPR 2024**](https://openaccess.thecvf.com/content/CVPR2024/papers/Jiang_Symphonize_3D_Semantic_Scene_Completion_with_Contextual_Instance_Queries_CVPR_2024_paper.pdf)

</div>

[![arXiv](https://img.shields.io/badge/arXiv-2306.15670-red?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2306.15670)
[![License: MIT](https://img.shields.io/github/license/hustvl/symphonies)](LICENSE)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/symphonize-3d-semantic-scene-completion-with/3d-semantic-scene-completion-from-a-single-1)](https://paperswithcode.com/sota/3d-semantic-scene-completion-from-a-single-1?p=symphonize-3d-semantic-scene-completion-with)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/symphonize-3d-semantic-scene-completion-with/3d-semantic-scene-completion-on-kitti-360)](https://paperswithcode.com/sota/3d-semantic-scene-completion-on-kitti-360?p=symphonize-3d-semantic-scene-completion-with)

**TL;DR:** Our paper delve into enhancing SSC through the utilization of instance-centric representations. We propose a novel paradigm that integrates ***instance queries*** to facilitate ***instance semantics*** and capture ***global context***. Our approach achieves SOTA results of ***15.04 & 18.58 mIoU*** on the SemanticKITTI & KITTI-360, respectively.

This project is built upon ***[TmPL](https://github.com/npurson/tmpl)***, a template for rapid & flexible DL experimentation development built upon [Lightning](https://lightning.ai/) & [Hydra](https://hydra.cc/).

![arch](assets/arch.png)

![vis](assets/vis.png)

![vis](assets/demo.gif)

## News

* ***Feb 27 '24***: **Our paper has been accepted at CVPR 2024. 🎉**
* ***Nov 22 '23***: We have updated our paper on [arXiv](https://arxiv.org/abs/2306.15670) with the latest results.
* ***Sep 18 '23***: We have achieved state-of-the-art results on the recently published SSCBench-KITTI-360 benchmark.
* ***Jun 28 '23***: We have released the [arXiv paper](https://arxiv.org/abs/2306.15670) of Symphonies.

## Preliminary

### Installation

1. Install PyTorch and Torchvision referring to https://pytorch.org/get-started/locally/.
2. Install MMDetection referring to https://mmdetection.readthedocs.io/en/latest/get_started.html#installation.
3. Install the rest of the requirements with pip.

    ```bash
    pip install -r requirements.txt
    ```

### Dataset Preparation

#### 1. Download the Data

**SemanticKITTI:** Download the RGB images, calibration files, and preprocess the labels, referring to the documentation of [VoxFormer](https://github.com/NVlabs/VoxFormer/blob/main/docs/prepare_dataset.md) or [MonoScene](https://github.com/astra-vision/MonoScene#semantickitti).

**SSCBench-KITTI-360:** Refer to https://github.com/ai4ce/SSCBench/tree/main/dataset/KITTI-360.

#### 2. Generate Depth Predications

**SemanticKITTI:** Generate depth predications with pre-trained MobileStereoNet referring to VoxFormer https://github.com/NVlabs/VoxFormer/tree/main/preprocess#3-image-to-depth.

**SSCBench-KITTI-360:** Follow the same procedure as SemanticKITTI but ensure to [adapt the disparity value](https://github.com/ai4ce/SSCBench/issues/8#issuecomment-1674607576).

### Pretrained Weights

The pretrained weight of MaskDINO can be downloaded [here](https://github.com/hustvl/Symphonies/releases/download/v1.0/maskdino_r50_50e_300q_panoptic_pq53.0.pth).

## Usage

0. **Setup**

    ```shell
    export PYTHONPATH=`pwd`:$PYTHONPATH
    ```

1. **Training**

    ```shell
    python tools/train.py [--config-name config[.yaml]] [trainer.devices=4] \
        [+data_root=$DATA_ROOT] [+label_root=$LABEL_ROOT] [+depth_root=$DEPTH_ROOT]
    ```

    * Override the default config file with `--config-name`.
    * You can also override any value in the loaded config from the command line, refer to the following for more infomation.
        * https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/
        * https://hydra.cc/docs/advanced/hydra-command-line-flags/
        * https://hydra.cc/docs/advanced/override_grammar/basic/

2. **Testing**

    Generate the outputs for submission on the evaluation server:

    ```shell
    python tools/test.py [+ckpt_path=...]
    ```

3. **Visualization**

    1. Generating outputs

        ```shell
        python tools/generate_outputs.py [+ckpt_path=...]
        ```

    2. Visualization

        ```shell
        python tools/visualize.py [+path=...]
        ```

## Results

1. **SemanticKITTI**

    |                    Method                    | Split |  IoU  | mIoU  |         Download         |
    | :------------------------------------------: | :---: | :---: | :---: | :----------------------: |
    | [Symphonies](symphonies/configs/config.yaml) | val   | 41.92 | 14.89 | [log](https://github.com/hustvl/Symphonies/releases/download/v1.0/semantic_kitti.log) / [model](https://github.com/hustvl/Symphonies/releases/download/v1.0/semantic_kitti_e25_miou0.1489.ckpt) |
    | [Symphonies](symphonies/configs/config.yaml) | test  | 42.19 | 15.04 | [output](https://github.com/hustvl/Symphonies/releases/download/v1.0/scoring_output.txt) |

2. **KITTI-360**

    |                    Method                    | Split |  IoU  | mIoU  |         Download         |
    | :------------------------------------------: | :---: | :---: | :---: | :----------------------: |
    | [Symphonies](symphonies/configs/config.yaml) | test  | 44.12 | 18.58 | [log](https://github.com/hustvl/Symphonies/releases/download/v1.0/kitti_360.log) / [model](https://github.com/hustvl/Symphonies/releases/download/v1.0/kitti_360_e26_miou0.1836.ckpt) |

## Citation

If you find our paper and code useful for your research, please consider giving this repo a star :star: or citing :pencil::

```BibTeX
@article{jiang2023symphonies,
      title={Symphonize 3D Semantic Scene Completion with Contextual Instance Queries},
      author={Haoyi Jiang and Tianheng Cheng and Naiyu Gao and Haoyang Zhang and Tianwei Lin and Wenyu Liu and Xinggang Wang},
      journal={CVPR},
      year={2024}
}
```

## Acknowledgements

The development of this project is inspired and informed by [MonoScene](https://github.com/astra-vision/MonoScene), [MaskDINO](https://github.com/IDEA-Research/MaskDINO) and [VoxFormer](https://github.com/NVlabs/VoxFormer). We are thankful to build upon the pioneering work of these projects.

## License

Released under the [MIT](LICENSE) License.
