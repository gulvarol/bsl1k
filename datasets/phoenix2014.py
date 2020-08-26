import math
import os
import pickle as pkl

import cv2
import numpy as np

from datasets.videodataset import VideoDataset

cv2.setNumThreads(0)


class PHOENIX2014(VideoDataset):
    def __init__(
        self,
        root_path="data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T",
        inp_res=224,
        resize_res=256,
        setname="train",
        scale_factor=0.1,
        num_in_frames=16,
        evaluate_video=False,
        hflip=0.5,
        stride=0.5,
        gpu_collation=False,
        assign_labels="auto",
    ):
        self.root_path = root_path
        self.setname = setname  # train, val or test
        self.gpu_collation = gpu_collation
        self.inp_res = inp_res
        self.resize_res = resize_res
        self.scale_factor = scale_factor
        self.num_in_frames = num_in_frames
        self.evaluate_video = evaluate_video
        self.hflip = hflip
        self.stride = stride
        self.assign_labels = assign_labels
        infofile = os.path.join(root_path, "info", "info.pkl")
        print(f"Loading {infofile}")
        data = pkl.load(open(infofile, "rb"))
        self.videos = [s.strip() for s in data["videos"]["name"]]

        other_class_ix = 1232
        self.classes = data["videos"]["gloss_ids"]
        replace_cnt = 0
        for i, seq in enumerate(self.classes):
            for j, gid in enumerate(seq):
                if gid == -1:
                    replace_cnt += 1
                    self.classes[i][j] = other_class_ix
        print(f"Replaced {replace_cnt} -1s with {other_class_ix}")
        with open(os.path.join(self.root_path, "info", "words.txt"), "r") as f:
            self.class_names = f.read().splitlines()

        self.class_names.append("1232 __OTHER__")

        self.video_folder = "videos"
        meta_key = self.video_folder
        if gpu_collation:
            # GPU collation requires all inputs to share the same spatial input size
            self.video_folder = "videos-resized-256fps-256x256"
        self.set_video_metadata(data, meta_key=meta_key, fixed_sz_frames=gpu_collation)

        self.train = list(np.where(np.asarray(data["videos"]["split"]) == 0)[0])
        if self.setname == "val":
            self.valid = list(np.where(np.asarray(data["videos"]["split"]) == 1)[0])
        elif self.setname == "test":
            self.valid = list(np.where(np.asarray(data["videos"]["split"]) == 2)[0])

        if self.assign_labels == "auto":
            self.frame_level_glosses = data["videos"]["alignments"]["gloss_id"]

        if evaluate_video:
            self.valid, self.t_beg = self._slide_windows(self.valid)

        VideoDataset.__init__(self)

    def _set_datasetname(self):
        self.datasetname = "phoenix2014"

    def _get_video_file(self, ind):
        return os.path.join(self.root_path, self.video_folder, self.videos[ind])

    def _get_sequence(self, ind):
        return self.classes[ind], len(self.classes[ind])

    def _get_class(self, ind, frame_ix):
        total_duration = self.num_frames[ind]
        t_middle = frame_ix[0] + (self.num_in_frames / 2)
        # Uniformly distribute the glosses over the video
        # auto labels are only for training
        if (
            self.assign_labels == "uniform"
            or self.setname != "train"
            or len(self.frame_level_glosses[ind]) == 0
        ):
            glosses = self.classes[ind]
            num_glosses = len(glosses)
            duration_per_gloss = total_duration / num_glosses
            glossix = math.floor(t_middle / duration_per_gloss)
            return glosses[glossix]
        # Use the automatic alignments
        elif self.assign_labels == "auto":
            frame_glosses = self.frame_level_glosses[ind]
            lfg = len(frame_glosses)

            # LABEL OF THE MIDDLE FRAME
            # t_middle might fall out of boundary
            # in that case pick the last frame
            # if lfg <= int(t_middle):
            #     t_middle = lfg - 1
            # glossix = frame_glosses[int(t_middle)]

            # DOMINANT LABEL WITHIN THE CLIP
            clip_glosses = [
                frame_glosses[i]
                for i in frame_ix
                if i < lfg
            ]
            clip_glosses = np.asarray(clip_glosses)
            glss, cnts = np.unique(clip_glosses, return_counts=True)
            # If there are multiple max, choose randomly.
            max_indices = np.where(cnts == cnts.max())[0]
            selected_max_index = np.random.choice(max_indices)
            return glss[selected_max_index]
        else:
            exit()

    def _get_nframes(self, ind):
        return self.num_frames[ind]

    def _get_img_width(self, ind):
        return self.img_widths[ind]

    def _get_img_height(self, ind):
        return self.img_heights[ind]
