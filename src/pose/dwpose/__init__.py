# https://github.com/IDEA-Research/DWPose
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
import time
import copy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
from controlnet_aux.util import HWC3, resize_image
from PIL import Image

from . import util
from .wholebody import Wholebody


def save_pose_info(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    foot = pose["foot"]

    body_score = pose['body_score']
    faces_score = pose["faces_score"]
    hands_score = pose["hands_score"]
    foot_score = pose["foot_score"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]

    save_info = {
        'W': W,
        'H': H,
        'bodies.candidate': np.around(candidate,4),
        'bodies.subset': np.around(subset,4),
        'bodies.score': np.around(body_score,2),
        'faces': np.around(faces,4),
        'hands': np.around(hands,4),
        'foot': np.around(foot,4),
        'faces_score': np.around(faces_score,2),
        'hands_score': np.around(hands_score,2),
        'foot_score': np.around(foot_score,2),
    }
    return save_info

def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]

    foot = pose["foot"]

    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    # canvas = util.draw_footpose(canvas, foot)

    canvas = util.draw_facepose(canvas, faces)

    return canvas

def draw_pose_new(pose,scores, H, W):
    bodies = pose["bodies"]
    hands = pose["hands"]
    foot = pose["foot"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]

    foot_score = scores['foot_score'].reshape(-1)
    hands_score = scores['hands_score'].reshape(-1)
    body_score = scores['body_score'].reshape(-1)

    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = util.draw_bodypose_new(canvas, candidate, subset, body_score)
    canvas = util.draw_handpose_new(canvas, hands, hands_score)
    canvas = util.draw_footpose_new(canvas, foot, foot_score)

    return canvas


class DWposeDetector:
    def __init__(self, model_root, foot = False):

        self.foot = foot
        self.model_root = model_root

    def to(self, device):
        self.pose_estimation = Wholebody(self.model_root, device)
        return self

    def cal_height(self, input_image):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image = HWC3(input_image)
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            # candidate[..., 0] /= float(W)
            # candidate[..., 1] /= float(H)
            body = candidate
        return body[0, ..., 1].min(), body[..., 1].max() - body[..., 1].min()

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        pose_type = 'classic',
        output_type="pil",
        **kwargs,
    ):
        # time_0 = time.time()
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image_org = HWC3(input_image)
        input_image = resize_image(input_image_org, detect_resolution)
        if pose_type != 'classic':
            H_new, W_new, C = input_image_org.shape
        H, W, C = input_image.shape

        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            score = subset[:, :18]
            max_ind = np.mean(score, axis=-1).argmax(axis=0)
            score = score[[max_ind]]
            body = candidate[:, :18].copy()
            body = body[[max_ind]]
            nums = 1
            body = body.reshape(nums * 18, locs)
            body_score = copy.deepcopy(score)
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1
            foot = candidate[[max_ind], 18:24]
            faces = candidate[[max_ind], 24:92]
            hands = candidate[[max_ind], 92:113]
            hands = np.vstack([hands, candidate[[max_ind], 113:]])

            foot_score = subset[[max_ind], 18:24]
            faces_score = subset[[max_ind], 24:92]
            hands_score = subset[[max_ind], 92:113]
            hands_score = np.vstack([hands_score, subset[[max_ind], 113:]])

            if self.foot:
                return foot, foot_score

            bodies = dict(candidate=body, subset=score)
            poseandscere = dict(bodies=bodies, hands=hands, faces=faces, foot=foot,
                        body_score=body_score,
                        hands_score=hands_score, 
                        faces_score=faces_score, 
                        foot_score=foot_score,
                    )
            save_dict = save_pose_info(poseandscere, H, W)
            # time_1 = time.time()

            pose = dict(bodies=bodies, hands=hands, faces=faces, foot=foot)
            
            if pose_type != 'classic':
                scores = dict(foot_score=foot_score, hands_score=hands_score,body_score=body_score)
                detected_map = draw_pose_new(pose, scores, H_new, W_new)
                
            else:
                detected_map = draw_pose(pose, H, W)

            detected_map = HWC3(detected_map)
            if pose_type != 'classic':
                detected_map = cv2.resize(
                    detected_map, (W_new, H_new), interpolation=cv2.INTER_LINEAR
                )       
            else:
                detected_map = cv2.resize(
                    detected_map, (W, H), interpolation=cv2.INTER_LINEAR
                )

            if output_type == "pil":
                detected_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

            # time_2 = time.time()
            return detected_map, body_score, save_dict # , [time_1 - time_0, time_2 - time_1]
