import json
import os
import pickle as pkl

import cv2
import numpy as np
import torch
from datasets.videodataset import VideoDataset
from utils.imutils import im_to_video, video_to_im

cv2.setNumThreads(0)


class BSL1K(VideoDataset):
    def __init__(
        self,
        info_pkl_json="misc/bsl1k/info-pkls.json",
        inp_res=224,
        resize_res=256,
        setname="train",
        scale_factor=0.1,
        num_in_frames=16,
        evaluate_video=False,
        hflip=0.5,
        stride=0.5,
        mouthing_prob_thres=0.9,
        gpu_collation=False,
        num_last_frames=20,
        featurize_mode=False,
        featurize_mask="",
        word_data_pkl=None,
        input_type="rgb",
        pose_keys=["body", "face", "lhnd", "rhnd"],
        mask_rgb=None,
        mask_type=None,
        bsl1k_pose_subset=False,
        bsl1k_anno_key="original-mouthings",
    ):
        self.setname = setname  # train, val or test
        self.featurize_mode = featurize_mode
        self.featurize_mask = featurize_mask
        self.gpu_collation = gpu_collation
        self.inp_res = inp_res
        self.resize_res = resize_res
        self.scale_factor = scale_factor
        self.num_in_frames = num_in_frames
        self.evaluate_video = evaluate_video
        self.hflip = hflip
        self.stride = stride
        self.input_type = input_type
        self.pose_keys = pose_keys
        self.mask_rgb = mask_rgb
        self.mask_type = mask_type

        assert self.num_in_frames == 16
        self.num_last_frames = num_last_frames
        print(f"Using only {self.num_last_frames} last frames of videos")

        with open(info_pkl_json, "r") as f:
            pkls = json.load(f)[bsl1k_anno_key]
        infofile = pkls["info"]

        self.video_folder = pkls["videos"]

        print(f"Loading {infofile}")
        data = pkl.load(open(infofile, "rb"))
        if self.input_type == "pose":
            pose_pkl = pkls["pose"]
            print(f"Loading {pose_pkl}")
            self.pose_data = pkl.load(open(pose_pkl, "rb"))
        if self.mask_rgb:
            assert bsl1k_pose_subset
            assert mask_type
        if self.mask_rgb == "face":
            face_pkl = pkls["face_bbox"]
            print(f"Loading {face_pkl}")
            self.face_data = pkl.load(open(face_pkl, "rb"))

        if bsl1k_pose_subset:  # self.mask_rgb:
            mouth_pkl = pkls["mouth_bbox"]
            print(f"Loading {mouth_pkl}")
            self.mouth_data = pkl.load(open(mouth_pkl, "rb"))

        self.set_video_metadata(data, meta_key="videos", fixed_sz_frames=gpu_collation)
        subset_ix = self.set_class_names(data=data, word_data_pkl=word_data_pkl)

        self.train = list(np.where(np.asarray(data["videos"]["split"]) == 0)[0])  # train
        self.valid = list(np.where(np.asarray(data["videos"]["split"]) == 2)[0])  # test
        self.videos = [s.strip() for s in data["videos"]["name"]]

        # Take subsets based on 'mouthing_prob'
        confident_mouthing = np.where(
            np.asarray(data["videos"]["mouthing_prob"]) >= mouthing_prob_thres
        )[0]
        msg = (
            f"Keeping {len(confident_mouthing)}/{len(data['videos']['mouthing_prob'])} "
            f"videos with more than {mouthing_prob_thres} mouthing confidence."
        )
        print(msg)
        self.train = [i for i in self.train if i in confident_mouthing]
        self.valid = [i for i in self.valid if i in confident_mouthing]

        print("Taking subsets according to word vocab")
        self.train = list(set(self.train).intersection(set(subset_ix)))
        self.valid = list(set(self.valid).intersection(set(subset_ix)))

        if self.input_type == "pose":
            valid_pose_ix = np.where(
                np.array([i is not None for i in self.pose_data["pose"]])
            )[0]
            print(f"{len(self.train)} train, {len(self.valid)} val samples.")
            print("Taking subsets according to having pose or not")
            self.train = list(set(self.train).intersection(set(valid_pose_ix)))
            self.valid = list(set(self.valid).intersection(set(valid_pose_ix)))
            print(f"{len(self.train)} train, {len(self.valid)} val samples.")

        if bsl1k_pose_subset:  # self.mask_rgb:
            # Valid mouth ix should be equivalent to valid face ix, so leaving this bit.
            valid_mouth_ix = np.where(
                np.array([i is not None for i in self.mouth_data])
            )[0]
            print(f"{len(self.train)} train, {len(self.valid)} val samples.")
            print("Taking subsets according to having pose or not")
            self.train = list(set(self.train).intersection(set(valid_mouth_ix)))
            self.valid = list(set(self.valid).intersection(set(valid_mouth_ix)))
            print(f"{len(self.train)} train, {len(self.valid)} val samples.")

        # Take a subset for validation if too large
        if self.setname == "val" and len(self.valid) > 1300:
            self.valid = self.valid[:: int(len(self.valid) / 1300)]

        if evaluate_video:
            self.valid, self.t_beg = self._slide_windows(self.valid)

        VideoDataset.__init__(self)

    def _set_datasetname(self):
        self.datasetname = "bsl1k"

    def _get_video_file(self, ind):
        return os.path.join(self.video_folder, self.videos[ind])

    def _get_class(self, ind, frame_ix=None):
        return self.classes[ind]

    def _get_nframes(self, ind):
        return self.num_frames[ind]

    def _get_img_width(self, ind):
        return self.img_widths[ind]

    def _get_img_height(self, ind):
        return self.img_heights[ind]

    def _get_pose(self, ind, frame_ix):
        part_dims = {"body": 18, "face": 70, "lhnd": 21, "rhnd": 21}
        # Total number of keypoints (e.g. 130 for all, 60 for body+hands)
        ndim = 0
        for k in self.pose_keys:
            ndim += part_dims[k]
        pose = torch.zeros((len(frame_ix), ndim, 3))
        ps = self.pose_data["pose"][ind]
        for i, f in enumerate(frame_ix):
            pose[i] = torch.cat([torch.Tensor(ps[f][k]) for k in self.pose_keys])
        # Normalize the xy values between 0,1 (pose was computed on the original 480x480 videos)
        pose[:, :, 0:2].div_(480)
        return pose.permute(2, 0, 1)  # 3 x 16 x ndims

    def _mask_rgb(self, rgb, ind, frame_ix, region="mouth", mask_type="exclude"):
        """
        frame_ix  : unused argument, because there is only 1 bbox for the whole video for now
        region    : mouth | face
        mask_type : exclude | include
        """
        assert rgb.ndim == 3
        # Convert from 48, 256, 256 -> 3, 16, 256, 256
        rgb = im_to_video(rgb)
        nframes = rgb.shape[1]
        res = rgb.shape[2]
        # Assuming square
        assert rgb.shape[2] == rgb.shape[3]
        if region == "mouth":
            region_bbox = self.mouth_data[ind]
        elif region == "face":
            region_bbox = self.face_data[ind]
        else:
            raise ValueError(f"Unknown region_bbox: {region_bbox}")

        region_bbox = (np.array(region_bbox) * res).round().astype(int)
        if mask_type == "exclude":
            region_pixels = rgb[
                :, :, region_bbox[0] : region_bbox[2], region_bbox[1] : region_bbox[3]
            ]
            mean_pixel = region_pixels.reshape(3, -1).mean(dim=1)
            for i in range(3):
                rgb[
                    i,
                    :,
                    region_bbox[0] : region_bbox[2],
                    region_bbox[1] : region_bbox[3],
                ] = mean_pixel[i]
        elif mask_type == "include":
            rgb[:, :, : region_bbox[0], :] = 0  # mean_pixel[i]
            rgb[:, :, region_bbox[2] :, :] = 0  # mean_pixel[i]
            rgb[:, :, :, : region_bbox[1]] = 0  # mean_pixel[i]
            rgb[:, :, :, region_bbox[3] :] = 0  # mean_pixel[i]
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")
        return video_to_im(rgb)
