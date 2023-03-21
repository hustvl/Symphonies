# Symphonies 🎻

[![](https://img.shields.io/github/license/npurson/symphonies)](LICENSE)

<!--
Refer to:

* https://github.com/open-mmlab/mmdetection/blob/3.x/projects/example_project/README.md
* https://github.com/open-mmlab/mmdetection/blob/3.x/configs/faster_rcnn/README.md
-->

***Symphonies (Scene-from-Insts)*** - Symphonize 3D Semantic Scene Completion with Instance Queries.

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
    python tools/train.py [--config-name config[.yaml]] [trainer.devices=4] [+data_root=$DATA_ROOT] [+preprocess_root=$PREPROCESS_ROOT]
    ```

    * Override the default config file with `--config-name`.
    * You can also override any value in the loaded config from the command line, refer to the following for more infomation.
        * https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/
        * https://hydra.cc/docs/advanced/hydra-command-line-flags/
        * https://hydra.cc/docs/advanced/override_grammar/basic/

2. **Generating outputs**

    ```shell
    python tools/generate_outputs.py [--config-name config[.yaml]] +model.ckpt_path=/path/to/ckpt
    ```

## License

Released under the [MIT](LICENSE) License.
