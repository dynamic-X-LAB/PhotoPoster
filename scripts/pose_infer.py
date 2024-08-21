import json
import requests
import io
import pickle
import base64
import numpy as np
from PIL import Image, ImageDraw
import argparse

from utils.pose_utils import pil2pose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    img = Image.open(args.image_path).convert('RGB')
    pose_pil, meta = pil2pose(img)
    pose_pil.save(args.image_path+'.pose.png')
    with open(args.image_path+'.pose.pkl', 'wb') as f:
        pickle.dump(meta, f)
