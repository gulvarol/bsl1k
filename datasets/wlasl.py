import os
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
import torch
from zsvision.zs_utils import memcache

from datasets.videodataset import VideoDataset
from utils.imutils import im_to_video, video_to_im

cv2.setNumThreads(0)


class WLASL(VideoDataset):
    def __init__(
        self,
        root_path="data/wlasl",
        inp_res=224,
        resize_res=256,
        setname="train",
        scale_factor=0.1,
        num_in_frames=64,
        evaluate_video=False,
        hflip=0.5,
        stride=0.5,
        ram_data=True,
        gpu_collation=False,
        use_bbox=True,
        monolithic_pkl_path="data/pickled-videos/wlasl-compressed-quality-90-resized-256x256.pkl",
        input_type="rgb",
        pose_keys=["body", "face", "lhnd", "rhnd"],
        mask_rgb=None,
        mask_type=None,
        mask_prob=1.0,
    ):
        self.root_path = root_path
        self.setname = setname  # train, val or test
        self.inp_res = inp_res
        self.resize_res = resize_res
        self.scale_factor = scale_factor
        self.num_in_frames = num_in_frames
        self.evaluate_video = evaluate_video
        self.hflip = hflip
        self.gpu_collation = gpu_collation
        self.stride = stride
        self.use_bbox = use_bbox
        self.input_type = input_type
        self.pose_keys = pose_keys
        self.mask_rgb = mask_rgb
        self.mask_type = mask_type

        self.video_folder = "videos_360h_25fps"
        if Path(monolithic_pkl_path).exists() and ram_data:
            print(f"Loading monolithic pickle from {monolithic_pkl_path}")
            self.video_data_dict = memcache(monolithic_pkl_path)
        else:
            self.video_data_dict = None

        infofile = os.path.join(root_path, "info", "info.pkl")
        print(f"Loading {infofile}")
        data = pkl.load(open(infofile, "rb"))

        if self.input_type == "pose":
            pose_pkl = os.path.join(root_path, "info", "pose.pkl")
            print(f"Loading {pose_pkl}")
            self.pose_data = pkl.load(open(pose_pkl, "rb"))
        if self.mask_rgb:
            assert mask_type
        if self.mask_rgb == "face":
            face_pkl = os.path.join(root_path, "info", "face_bbox.pkl")
            print(f"Loading {face_pkl}")
            self.face_data = pkl.load(open(face_pkl, "rb"))

        # Use this to take subset
        if self.input_type == "pose" or self.mask_rgb:
            mouth_pkl = os.path.join(root_path, "info", "mouth_bbox.pkl")
            print(f"Loading {mouth_pkl}")
            self.mouth_data = pkl.load(open(mouth_pkl, "rb"))

        self.videos = [s.strip() for s in data["videos"]["name"]]
        self.videos = np.asarray(self.videos)

        self.classes = data["videos"]["word_id"]
        with open(os.path.join(self.root_path, "info", "words.txt"), "r") as f:
            self.class_names = f.read().splitlines()

        meta_key = self.video_folder
        if gpu_collation and not self.video_data_dict:
            # GPU collation requires all inputs to share the same spatial input size
            self.video_folder = "videos-resized-256fps-256x256"
        self.set_video_metadata(data, meta_key=meta_key, fixed_sz_frames=gpu_collation)

        bboxes_orig = [s for s in np.asarray(data["videos"]["box"])]
        self.bboxes = []
        for i, bb in enumerate(bboxes_orig):
            ht = data["videos"]["videos_original"]["H"][i]
            wt = data["videos"]["videos_original"]["W"][i]
            xmin, ymin, xmax, ymax = bb
            bb_norm = [ymin / ht, xmin / wt, ymax / ht, xmax / wt]
            self.bboxes.append(bb_norm)

        self.train = list(np.where(np.asarray(data["videos"]["split"]) == 0)[0])
        if self.setname == "val":
            self.valid = list(np.where(np.asarray(data["videos"]["split"]) == 1)[0])
        elif self.setname == "test":
            self.valid = list(np.where(np.asarray(data["videos"]["split"]) == 2)[0])

        if self.input_type == "pose" or self.mask_rgb:
            # Valid mouth ix should be equivalent to valid face ix, valid pose ix etc
            valid_mouth_ix = np.where(
                np.array([i is not None for i in self.mouth_data])
            )[0]
            if self.setname == "val" or self.setname == "test":
                print(f"{len(self.train)} train, {len(self.valid)} val samples.")
            print("Taking subsets according to having pose or not")
            self.train = list(set(self.train).intersection(set(valid_mouth_ix)))
            if self.setname == "val" or self.setname == "test":
                self.valid = list(set(self.valid).intersection(set(valid_mouth_ix)))
                print(f"{len(self.train)} train, {len(self.valid)} val samples.")

        if evaluate_video:
            self.valid, self.t_beg = self._slide_windows(self.valid)

        VideoDataset.__init__(self)

    def _set_datasetname(self):
        self.datasetname = "wlasl"

    def _get_video_file(self, ind):
        return os.path.join(self.root_path, self.video_folder, self.videos[ind])

    def _get_class(self, ind, frame_ix=None):
        return self.classes[ind]

    def _get_nframes(self, ind):
        return self.num_frames[ind]

    def _get_img_width(self, ind):
        return self.img_widths[ind]

    def _get_img_height(self, ind):
        return self.img_heights[ind]

    def _get_bbox(self, ind):
        return self.bboxes[ind]

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
        # Normalize the xy values between 0,1 (pose was computed on the 256x256 resized videos)
        pose[:, :, 0:2].div_(256)
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
