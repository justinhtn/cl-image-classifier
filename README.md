# CL Img Classifier
Command line image classifier written in Pytorch. Requires category to label mapping json file. See 'cat_to_name' json for default specs.


## Running training script

`python train.py 'path to data directory'`

#### Train cl arguments options and defaults

Data directory: `--save_path` default='./flowers/'

Learning rate: `--lr` default=0.003

Hidden layers: `--hl` default=1000

Device: `--device ` default= cuda

Save path: `--save_path` default=./checkpoint.pth

## Running predict scripy

`python predict.py 'path to output directory'`

#### Predict cl arguments options and defaults

Model checkpoint path: `--checkpoint_path` default='./checkpoint.pth'

Image to predict path: `--image_path` default='flowers/test/1/image_06760.jpg'

K likely classes to return: `--k` default=1

Path to category mappings json: `--categories` default='cat_to_name.json'
