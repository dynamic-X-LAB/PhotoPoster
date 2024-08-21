import json
import requests
import io
import pickle
import base64
import numpy as np
from PIL import Image, ImageDraw


import os, sys
pwd=os.getcwd()
print(pwd)
sys.path.append(pwd)


def pil2pose(pil):
    url = "http://127.0.0.1:8001/process-image/"
    imgio = io.BytesIO()
    pil.save(imgio, format='JPEG', quality=95)
    image_bytes = base64.b64encode(imgio.getvalue()).decode()
    data = {'image': image_bytes}
    response = requests.post(url, json=data)
    json_response = response.json()
    info_content = base64.b64decode(json_response["info"])
    pil_content = base64.b64decode(json_response["pil"])
    pil = Image.open(io.BytesIO(pil_content))
    info = pickle.load(io.BytesIO(info_content))
    return pil, info

def get_hand_meta(hands, hands_score, w, h, hidx=0, margin=0.2, conf=5):
    hands1 = hands[hidx]
    hands_score1 = hands_score[hidx*16:hidx*16+16]
    hands1 = hands1[hands_score1>conf]
    if len(hands1)<8:
        return None
    xmin, ymin = hands1.min(axis=0)
    xmax, ymax = hands1.max(axis=0)
    wd, hd = xmax-xmin, ymax-ymin
    std1 = hands1.std(axis=0)
    xmin = max(int(xmin-margin*wd), 0)
    xmax = min(int(xmax+margin*wd), w)
    ymin = max(int(ymin-margin*hd), 0)
    ymax = min(int(ymax+margin*hd), h)
    bbox = [xmin,ymin,xmax,ymax, std1[0].item(), std1[1].item()]
    return bbox

def crop_hand(img, meta, hidx, crop_ratio, new_size, margin=0.2):
    img = np.array(img) if not isinstance(img, np.ndarray) else img
    w,h = meta['W'], meta['H']
    hands = meta['hands']
    body = meta['bodies.candidate2']
    hands_score = meta['hands_score']
    hand_bbox = get_hand_meta(hands, hands_score, w, h, hidx=hidx, margin=margin)
    body2hand = {0:7, 1:4}
    if hand_bbox is None:
        center = body[body2hand[hidx]]
        width = (meta['bodies.candidate2'][5][0] - meta['bodies.candidate2'][2][0])//3
        x1,y1,x2,y2 = center[0]-width, center[1]-width, center[0]+width, center[1]+width
        x1 = int(max(x1, 0))
        y1 = int(max(y1, 0))
        x2 = int(min(x2, w))
        y2 = int(min(y2, h))
        guider_img = img.copy()
        hand_vis_flag = False
        hand_bbox = [x1,y1,x2,y2]
    else:
        x1,y1,x2,y2 = hand_bbox[:4]
        center = [(x1+x2)//2, (y1+y2)//2]
        guider_img = img.copy()
        hand_vis_flag = True
    x,y = center
    crop_size = max(x2-x1, y2-y1) * crop_ratio
    x1 = max(x-crop_size//2, 1)
    y1 = max(y-crop_size//2, 1)
    x2 = min(x+crop_size//2, w-1)
    y2 = min(y+crop_size//2, h-1)
    x1, y1 = int(max(x1, 0)), int(max(y1, 0))
    x2, y2 = int(min(x2, w)), int(min(y2, h))
    guider_img = Image.fromarray(guider_img[y1:y2, x1:x2, :]).resize((new_size, new_size))
    crop_img = Image.fromarray(img[y1:y2, x1:x2, :]).resize((new_size, new_size))
    crop_bbox = [x1,y1,x2,y2]
    wratio = new_size / (x2-x1)
    hratio = new_size / (y2-y1)
    new_hand_bbox = [
                        int((hand_bbox[0]-x1)*wratio),
                        int((hand_bbox[1]-y1)*hratio),
                        int((hand_bbox[2]-x1)*wratio),
                        int((hand_bbox[3]-y1)*hratio)
                       ]
    return crop_img, guider_img, hand_vis_flag, new_hand_bbox, crop_bbox

def combine_images(image_list, border_size=10, rownum=8, border_color=(255, 255, 255)):
    image_list = [Image.fromarray(i) if isinstance(i, np.ndarray) else i for i in image_list ]
    img_width, img_height = image_list[0].size
    total_width = img_width * rownum + border_size * 5 + 30
    total_height = ((len(image_list) - 1) // rownum + 1) * (img_height + border_size) + border_size + 30
    new_img = Image.new('RGB', (total_width, total_height), border_color)
    draw = ImageDraw.Draw(new_img)
    for i, img in enumerate(image_list):
        row = i // rownum
        col = i % rownum
        x = col * (img_width + border_size) + border_size
        y = row * (img_height + border_size) + border_size
        new_img.paste(img, (x, y))
    return new_img





if __name__ == '__main__':
    from tqdm import tqdm
    image_paths = ['../assets/stage1/ref_img1.png', '../assets/stage1/ref_img2.png', 
                   '../assets/stage1/pose_img1.png', '../assets/stage1/pose_img2.png']
    for image_path in tqdm(image_paths):
        img = Image.open(image_path).convert('RGB')
        pose_pil, meta = pil2pose(img)
        pose_pil.save(image_path+'.pose.png')
        with open(image_path+'.pose.pkl', 'wb') as f:
            pickle.dump(meta, f)
