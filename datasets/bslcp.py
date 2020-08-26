import json
import os
import pickle as pkl

import cv2
import numpy as np

from datasets.videodataset import VideoDataset

cv2.setNumThreads(0)


class BSLCP(VideoDataset):
    def __init__(
        self,
        root_path="data/BSLCP",
        inp_res=224,
        resize_res=256,
        setname="train",
        scale_factor=0.1,
        num_in_frames=16,
        evaluate_video=False,
        hflip=0.5,
        stride=0.5,
        gpu_collation=False,
        word_data_pkl=None,
        featurize_mask="",
        featurize_mode=False,
    ):
        self.root_path = root_path
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

        infofile = os.path.join(root_path, "info/info.pkl")
        self.video_folder = "videos-resized-25fps-256x256-signdict_signbank"

        print(f"Loading {infofile}")
        data = pkl.load(open(infofile, "rb"))

        self.set_video_metadata(data, meta_key="videos", fixed_sz_frames=gpu_collation)
        self.set_class_names(data=data, word_data_pkl=word_data_pkl)

        self.train = list(np.where(np.asarray(data["videos"]["split"]) == 0)[0])
        self.valid = list(np.where(np.asarray(data["videos"]["split"]) == 2)[0])

        self.videos = [s.strip() for s in data["videos"]["name"]]

        if evaluate_video:
            self.valid, self.t_beg = self._slide_windows(self.valid)

        VideoDataset.__init__(self)

    def _set_datasetname(self):
        self.datasetname = "BSLCP"

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
