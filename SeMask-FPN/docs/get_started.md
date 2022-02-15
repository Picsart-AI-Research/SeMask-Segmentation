## Prerequisites

- Linux or macOS
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

> Note: You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

### A from-scratch setup script

Here is a full script for setting up mmsegmentation with conda and link the dataset path (supposing that your dataset path is $DATA_ROOT).

```shell
conda create -n semask python=3.7 -y
conda activate semask

conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
pip install mmcv-full==latest+torch1.5.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
git clone https://github.com/Picsart-AI-Research/SeMask-Segmentation
cd SeMask-Segmentation/SeMask-FPN
pip install -e .  # or "python setup.py develop"

mkdir data
ln -s $DATA_ROOT data
```

### Note:

1. The `version+git_hash` will also be saved in trained models meta, e.g. 0.5.0+c415a2e.
2. When MMsegmentation is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it.
3. If you would like to use `opencv-python-headless` instead of `opencv-python`,
   you can install it before installing MMCV.
4. Some dependencies are optional. Simply running `pip install -e .` will only install the minimum runtime requirements.
   To use optional dependencies like `cityscapessripts`  either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.