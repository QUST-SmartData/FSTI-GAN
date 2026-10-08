import os
import torch
import argparse
import random
import numpy as np
import yaml
import shutil 
from src.config import Config
from src.texture_flow import TextureFlow

def main(mode=None):
    r"""starts the model
    Args:
        mode : train, test, eval, reads from config file if not specified
    """

    config = load_config(mode)
    config.MODE = mode
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    seed = int(config.SEED or 12)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        #   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    model = TextureFlow(config)
    
    if mode == 'train':
        config.print()
        print('\nstart training...\n')
        model.train()

    elif mode == 'test':
        print('\nstart test...\n')
        model.test()

    elif mode == 'eval':
        print('\nstart eval...\n')
        model.eval()        


def load_config(mode=None):
    r"""loads model config 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='fstigan', help='output model name.')
    parser.add_argument('--config', type=str, default='model_config.yaml', help='Path to the config file.')
    parser.add_argument('--path', type=str, default='./results', help='outputs path')
    parser.add_argument("--resume_all", action="store_true", help='load model from checkpoints')
    parser.add_argument("--remove_log", action="store_true", help='remove previous tensorboard log files')


    if mode == 'test':
        parser.add_argument('--input', type=str, help='path to the input image files')
        parser.add_argument('--mask', type=str, help='path to the mask files')
        parser.add_argument('--structure', type=str, help='path to the structure files')
        parser.add_argument('--output', type=str, help='path to the output directory')
        parser.add_argument('--model', type=int, default=1, help='which model to test')
    
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sr-checkpoint', type=str, default=None)
    parser.add_argument('--tr-checkpoint', type=str, default=None)
    opts = parser.parse_args()
    config = Config(opts, mode)
    output_dir = os.path.join(opts.path, opts.name)
    perpare_sub_floder(output_dir)

    if mode == 'train':
        config_dir = os.path.join(output_dir, 'config.yaml')
        with open(config_dir, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config._dict, f, sort_keys=False, allow_unicode=True)
    return config


def perpare_sub_floder(output_path):
    img_dir = os.path.join(output_path, 'images')
    if not os.path.exists(img_dir):
        print("Creating directory: {}".format(img_dir))
        os.makedirs(img_dir)     


    checkpoints_dir = os.path.join(output_path, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        print("Creating directory: {}".format(checkpoints_dir))
        os.makedirs(checkpoints_dir) 
    

if __name__ == "__main__":
    main()
