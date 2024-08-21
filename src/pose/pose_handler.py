import numpy as np
from mmpose.apis import MMPoseInferencer
import os
from omegaconf import OmegaConf
import numpy as np
from .dwpose import DWposeDetector
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from utils.utils_tools import draw_pose, get_info

class Handler:
    def __init__(self, config_file) -> None:
        config = OmegaConf.load(config_file)
        self.cfg = config
        self.inferencer_body = MMPoseInferencer(
            pose2d=config.Pose_set.body_set,
            pose2d_weights=config.Pose_set.body_model_path,
            det_model=config.Pose_set.body_det_model,
            det_weights=config.Pose_set.body_det_weights,
            det_cat_ids=[0],
        )
        self.inferencer_hand = MMPoseInferencer(
            pose2d=config.Pose_set.hand_set,
            pose2d_weights=config.Pose_set.hand_model_path,
            det_model=config.Pose_set.body_det_model,
            det_weights=config.Pose_set.body_det_weights,
            det_cat_ids=[0],
        )
        self.inferencer_foot = DWposeDetector(config.Pose_set.dwpose_root, foot=True).to("cuda")
        

    def predict(self, frame, draw_hand_flag=[1,1]):
        # Frame is BGR Mode
        H, W = frame.shape[0], frame.shape[1]
        body = self.inferencer_body(frame)
        body = next(body)['predictions'][0][0]
        body_w = self.inferencer_hand(frame)
        hand = next(body_w)['predictions'][0][0]
        feet, feet_scores = self.inferencer_foot(frame[:,:,::-1])
        feet = feet.tolist()[0]
        for points in feet:
            points[0] = points[0] * W
            points[1] = points[1] * H
        foot = {
            'keypoints': feet,
            'keypoint_scores': feet_scores.tolist()[0],
        }

        info = get_info(body, hand, foot, W, H)
        detected_map_pil = None
        detected_map_pil = draw_pose(info, draw_hand_flag)
        info = {
            'W': info['W'],
            'H': info['H'],
            'bodies.candidate': np.around(info['body_points'],4),
            'bodies.score':np.around(info['body_scores'],2),
            'hands':np.around(info['hands_left']+info['hands_right'],4),
            'hands_score':np.around(info['hands_left_score']+info['hands_right_score'],2),
            'foot':np.around(info['feet_left']+info['feet_right'],4),
            'foot_score':np.around(info['feet_left_score']+info['feet_right_score'],2),
            'face': info['face'],
            'face_score': info['face_score'],
        }

        out_info = {'W': info['W'], 'H': info['H']}
        for key in ['foot', 'faces', 'hands', "bodies.candidate", "bodies.subset","foot_score","hands_score", "bodies.score"]:
            if key not in info.keys():
                continue
            value = info[key]
            if not(key == "bodies.subset" or key == "foot_score" or key == "hands_score" or key == "bodies.score"):
                value = value.reshape(-1,2)
                if key == "hands": value = np.array([value[0:16], value[16:32]])
            out_info[key] = value
        index_map = [0,17,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3]
        out_info['bodies.candidate2'] = np.take(out_info['bodies.candidate'], index_map, axis=0)
        out_info['bodies.score2'] = np.take(out_info['bodies.score'], index_map, axis=0)
        out_info['bodies.candidate'] = np.take(out_info['bodies.candidate'], index_map, axis=0) / np.array([info['W'], info['H']])
        if 'bodies.score' in out_info.keys():
            out_info['bodies.score'] = np.take(out_info['bodies.score'], index_map, axis=0)
        else:
            out_info['bodies.score'] = np.ones([len(index_map)], dtype=np.float32)
        return out_info, detected_map_pil 

if __name__ == '__main__':
    from tqdm import tqdm
    handler = Handler()
