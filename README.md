# Symphonies (Scene-from-Insts) 🎻

[![](https://img.shields.io/github/license/npurson/symphonies)](LICENSE)

<!--
Refer to:

* https://github.com/open-mmlab/mmdetection/blob/3.x/projects/example_project/README.md
* https://github.com/open-mmlab/mmdetection/blob/3.x/configs/faster_rcnn/README.md
-->

### **[Symphonize 3D Semantic Scene Completion with Instance Queries](TODO)**, NeurIPS 2023 Submission

[Haoyi Jiang](https://github.com/npurson)<sup>1</sup>,
[Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ)<sup>1</sup>,
Naiyu Gao<sup>2</sup>,
Haoyang Zhang<sup>2</sup>,
[Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>,
[Xinggang Wang](https://xwcv.github.io/)<sup>1,✉️</sup>
<br>
<sup>1</sup>[School of EIC, HUST](http://english.eic.hust.edu.cn/),
<sup>2</sup>[Horizon Robotics](https://en.horizonrobotics.com/)

***TL;DR:*** We delve into a novel query-based paradigm for SSC that not only aims to enhance performance, but also ensures compatibility with large vision models as encoders and accommodating the requirements of modern query-based multi-task autonomous driving systems. Our approach achieves 12 mIoU & 37.7 IoU on the SemanticKITTI benchmark.

This project is built upon ***[TmPL](https://github.com/npurson/tmpl)***, a template for rapid & flexible DL development with [Lightning](https://lightning.ai/) & [Hydra](https://hydra.cc/).

## Installation

1. Install PyTorch and Torchvision referring to https://pytorch.org/get-started/locally/.
2. Install MMDetection referring to https://mmdetection.readthedocs.io/en/latest/get_started.html#installation.
3. Install the rest of the requirements with pip.

    ```bash
    pip install -r requirements.txt
    ```

## Usage

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

2. **Generating outputs**

    ```shell
    python tools/generate_outputs.py [--config-name config[.yaml]] [+model.ckpt_path=/path/to/ckpt]
    ```

## License

Released under the [MIT](LICENSE) License.
