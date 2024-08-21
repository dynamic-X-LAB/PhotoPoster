#!/usr/bin/env python
#-*- coding:utf8 -*-
import os, sys
import os.path as osp
import traceback
from PIL import Image
import cv2
import numpy as np
from skimage import transform as trans

def get_similarity_M(src_points, dst_points):
    tform = trans.SimilarityTransform()
    tform.estimate(src_points, dst_points)
    M = tform.params[0:2, :]
    return M

def affine_on_landmarks(affine_matrix, pts):
    """
    Do affine transform on landmarks
    :param affine_matrix: Affine matrix
    :param pts: Landmarks
    :return:
    """
    if (pts is None or len(pts) <=0):
        return np.array([])
    pts_one = np.ones([pts.shape[0], 3])
    pts_one[:, 0:2] = pts
    pts_trans = affine_matrix.dot(pts_one.transpose())
    return pts_trans.transpose()


def process(ref_image_pil, ref_image_pts, pose_images_pil, pose_video_pts, dst_size=(512, 768)):
    # ref image align to dst_size
    ref_img = np.array(ref_image_pil)    
    h, w, _ = ref_img.shape
    src_pts = np.array([[w/2.0, 0], [w/2.0, h-1]])
    dst_w, dst_h = dst_size
    dst_pts = np.array([[dst_w/2.0, 0], [dst_w/2.0, dst_h-1]])

    M = get_similarity_M(src_pts, dst_pts)
    ref_img_align = cv2.warpAffine(ref_img, M, dst_size, borderValue=(255, 255, 255))
    pts = ref_image_pts[0]["bodies.candidate"] * np.array([w, h])

    ref_pts_align = affine_on_landmarks(M, pts)

    # pose video align to ref image
    w = pose_video_pts[0]['W']
    h = pose_video_pts[0]['H']

    pose_video_pts_0 = pose_video_pts[0]["bodies.candidate"] * np.array([w, h]) 
    pts_index = [0, 2, 1, 5, 11, 8]

    src_pts = pose_video_pts_0[pts_index]
    dst_pts = ref_pts_align[pts_index]

    x0, y0, x1, y1= src_pts[:, 0].min(), src_pts[:, 1].min(), src_pts[:, 0].max(), src_pts[:, 1].max()
    xx0, yy0, xx1, yy1= dst_pts[:, 0].min(), dst_pts[:, 1].min(), dst_pts[:, 0].max(), dst_pts[:, 1].max()

    src_pts = np.array( [[(x0+x1)*0.5, y0],[(x0+x1)*0.5, y1]])
    dst_pts = np.array( [[(xx0+xx1)*0.5, yy0],[(xx0+xx1)*0.5, yy1]])

    M = get_similarity_M(src_pts, dst_pts)
    pose_imaegs_align = []
    for img_pil in pose_images_pil:
        pose_img = np.array(img_pil)
        pose_img_align = cv2.warpAffine(pose_img, M, dst_size, borderValue=(0, 0, 0))
        pose_img_align = Image.fromarray(pose_img_align)
        pose_imaegs_align.append(pose_img_align)
    
    return Image.fromarray(ref_img_align), pose_imaegs_align

