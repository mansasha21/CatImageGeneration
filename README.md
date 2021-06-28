# CatImageGeneration

## Prerequisites
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
- Create env `conda create --name pix2pix'
- Install requirements `pip install -r requirements.txt`.

### train/test
- To download pretrained model and dataset run `bash bin/download.sh`

- Train a model:
`python train.py --train_dir ../data_examples/train/ --device cuda --checkpoint_dir ../models/`
 You can also set lr, max_epochs, l1_coef, checkpoint_freq, and val_dir

- Generate:
`python generate.py --edge_dir .../data_examples/generate/ --device cuda --gen_path PATH_TO_PRETRAINED_MODEL --save_dir ../data_examples/generated_imgs/`

