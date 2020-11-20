# BSL-1K: Scaling up co-articulated sign language recognition using mouthing cues

[Samuel Albanie*](https://www.robots.ox.ac.uk/~albanie/), [Gül Varol*](http://www.di.ens.fr/~varol/), [Liliane Momeni](http://www.robots.ox.ac.uk/~liliane/), [Triantafyllos Afouras](http://www.robots.ox.ac.uk/~afourast/), [Joon Son Chung](https://joonson.com/), [Neil Fox](https://www.ucl.ac.uk/dcal/people/research-staff/neil-fox) and [Andrew Zisserman](https://www.robots.ox.ac.uk/~az/),
*BSL-1K: Scaling up co-articulated sign language recognition using mouthing cues*, ECCV 2020.

[[Project page]](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/) [[arXiv]](https://arxiv.org/abs/2007.12131)

<!-- <p align="center">
<img src="http://www.robots.ox.ac.uk/~vgg/research/bsl1k/images/bsl1k.gif"/>
</p> -->

## Contents
* [Setup](https://github.com/gulvarol/bsl1k#setup)
* [Demo](https://github.com/gulvarol/bsl1k#demo)
* [Train and Test](https://github.com/gulvarol/bsl1k#train-and-test)
  * [Datasets](https://github.com/gulvarol/bsl1k#datasets)
  * [Pretrained models](https://github.com/gulvarol/bsl1k#pretrained-models)
  * [Train](https://github.com/gulvarol/bsl1k#train)
  * [Test](https://github.com/gulvarol/bsl1k#test)
* [Experiments](https://github.com/gulvarol/bsl1k#experiments)
  * [Experiments on BSL-1K](https://github.com/gulvarol/bsl1k#experiments-on-bsl-1k)
  * [Experiments on Transfer](https://github.com/gulvarol/bsl1k#experiments-on-transfer)
* [Note on BSL-1K data release](https://github.com/gulvarol/bsl1k#note-on-bsl-1k-data-release)
* [Limitations](https://github.com/gulvarol/bsl1k#limitations)
* [Citation](https://github.com/gulvarol/bsl1k#citation)

## Setup

**Requires: python 3.6.**  (*some non-essential pre-processing scripts require python 3.7*)

``` bash
# Clone this repository
git clone https://github.com/gulvarol/bsl1k.git
cd bsl1k/
# Setup symbolic links (point these to folders where you would like data and checkpoints to be stored)
ln -s <replace_with_data_path> data
ln -s <replace_with_log_path> checkpoint
# Create bsl1k_env environment with dependencies
conda env create -f environment.yml
conda activate bsl1k_env
pip install -r requirements.txt
```

## Demo
The `demo` folder contains a sample script to apply sign language recognition on an input video.  By default, the demo will download: (1) a model that has been pretrained on BSL-1K and then fine-tuned on [WLASL](https://arxiv.org/abs/1910.11006), (2) a video from [handspeak.com](http://handspeak.com/) (this particular video is part of the the WLASL test set).  The demo should produce the output below (you can change to other inputs):

Usage: run `python demo.py`.

<!-- ![book sign example](demo/sample_data/demo-output.gif) -->
<img src="demo/sample_data/demo-output.gif" alt="ASL example for book" height="300px">

The original video source can be found [here](https://www.handspeak.com/word/search/index.php?id=210). Copyright Jolanta Lapiak.

## Train and Test

### Supported Datasets
* This code supports I3D classification training for the following sign language video datasets:

| Dataset      | `--datasetname`        | Path                                             | `--num-classes`          | `--ram_data` | `info/`   |
|--------------|------------------------|--------------------------------------------------|--------------------------|--------------|--------------|
| BSL-1K (coming soon)      | `bsl1k`                | `data/bsl1k/`                                    | 1064                     | 0            | [COMING SOON] |
| WLASL        | `wlasl`                | `data/wlasl/`                                    | 2000                     | 1            | [(3.7GB)](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/info/wlasl.tar) |
| MSASL        | `msasl`                | `data/msasl/`                                    | 1000                     | 1            | [(6.6GB)](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/info/msasl.tar) |
| Phoenix2014T | `phoenix2014`          | `data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/` | 1233                     | 0            | [(3MB)](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/info/phoenix2014t.tar) |
| BSL-Corpus   | `bslcp`                | `data/BSLCP/`                                    | 966                      | 0            | [(1MB)](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/info/bslcp.tar) |

* Please cite the original papers for [WLASL](https://github.com/dxli94/WLASL), [MSASL](https://www.microsoft.com/en-us/research/project/ms-asl/), [Phoenix2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) and [BSL-Corpus](https://bslcorpusproject.org/cava/) datasets. Here, we only provide pre-processed metadata, but not the videos, which can instead be obtained via the metadata provided by the dataset authors, as described next:

### Preparing the data
<u>**WLASL**</u>: First head to the WLASL authors' github page [here](https://github.com/dxli94/WLASL) and download the `.json` file of links. This file evolves over time, the current version is v3 and is called `WLASL_v0.3.json` .  Place the downloaded file at the location `data/wlasl/info/WLASL_v0.3.json`.  After this step, video files can be downloaded by running the following command:

```python misc/wlasl/download_wlasl.py```
  
*Notes*: some videos may no longer be accessible - you can contact the WLASL authors to address this issue (they provide an email address on the github page linked above). Also note that the v3 json may produce slightly different results from the `WLASL_v0.1.json` we used for our experiments.

<u>**MSASL**</u>: As for the dataset above, first download the `json` files of video links from the authors [here](https://www.microsoft.com/en-us/research/project/ms-asl/) and place them into `data/msasl/info/`. This will create a file directory structure as follows:
```
data/
   msasl/
       info/
          MSASL_train.json
          MSASL_val.json
          MSASL_test.json
```
The videos may then be downloaded via:

```python misc/msasl/download_msasl.py```


<u>**Phoenix2014T**</u>: video files can be downloaded from [here](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) (this file should be unpacked to the location `data/PHOENIX-2014-T-release-v3`). You can then run the following command script to create `.mp4` video files from the provided `.png` frames:

```python misc/phoenix2014/gather_frames.py```

<u>**BSL-Corpus**</u>: can be downloaded from [here](https://bslcorpusproject.org/cava/) upon request from the owners.

* In our folder organization, each dataset has a subfolder `info/` in which most pre-extracted annotations are kept:
  * `info/info.pkl`
  * `info/pose.pkl` [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) is extracted for:
    * `bsl1k` for a subset of the videos
    * `msasl` and `wlasl` for all videos (we provide these within the `.tar` files)
* We have pre-processed all of the video datasets to be at 256x256 spatial resolution. The pre-processing scripts can be found under the `misc` folder for each dataset. Using the original videos is possible, but is slower.
* We have pre-processed WLASL and MSASL such that the video frames are stored in a pkl file, we then loaded the entire dataset in RAM. Setting `--ram_data` to 0 will not require this preprocessing step, and use the video files instead. The results are similar with and without this step.

### Pretrained models

You can download some of the pretrained models used in the experiments by running
`bash misc/pretrained_models/download.sh` in the project root directory. All the other pretrained models from the experiments are provided in the [Experiments](https://github.com/gulvarol/bsl1k#experiments) section. The best BSL-1K model reported for the final experiments is [the first model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m5_l20_kws8_ppose/model.pth.tar).

### Train
The training launch for each experiment can be found in the [Experiments](https://github.com/gulvarol/bsl1k#experiments) section by clicking "run" links. The training can be ran by directly
typing `python main.py <args>` on terminal with the arguments. We also provide the `exp/create_exp.py`
script that we used when launching experiments. You can use that via:
``` bash
cd exp/
# Change config.json contents
python train.py
```

### Test
``` bash
cd exp/
# Change config.json contents
python test.py
```

## Experiments

### Experiments on BSL-1K
* Best model BSL-1K(m.5), last 20 frames, video pose pretrained

| Model | ins. top-1 | ins. top-5 | cls. top-1 | cls. top-5 | Links |
| - | - | - | - | - | - |
| BSL-1K | 75.51 | 88.83 | 52.76 | 72.14 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m5_l20_kws8_ppose/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m5_l20_kws8_ppose/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m5_l20_kws8_ppose/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m5_l20_kws8_ppose/log.json) |

* Trade-off between training noise vs. size

| Model | ins. top-1 | ins. top-5 | cls. top-1 | cls. top-5 | Links |
| - | - | - | - | - | - |
| BSL-1K(m.5) | 70.61 | 85.26 | 47.47 | 68.13 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m5_l24_kws8_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m5_l24_kws8_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m5_l24_kws8_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m5_l24_kws8_pkinetics/log.json) |
| BSL-1K(m.6) | 71.33 | 85.92 | 48.83 | 68.82 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m6_l24_kws8_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m6_l24_kws8_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m6_l24_kws8_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m6_l24_kws8_pkinetics/log.json) |
| BSL-1K(m.7) | 70.95 | 85.73 | 48.13 | 67.81 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m7_l24_kws8_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m7_l24_kws8_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m7_l24_kws8_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m7_l24_kws8_pkinetics/log.json) |
| BSL-1K(m.8) | 69.00 | 83.79 | 45.86 | 64.42 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/log.json) |
| BSL-1K(m.9) | 60.53 | 77.51 | 35.09 | 54.26 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m9_l24_kws8_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m9_l24_kws8_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m9_l24_kws8_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m9_l24_kws8_pkinetics/log.json) |

* Contribution of individual cues (pose subset of the data)

| Model | ins. top-1 | ins. top-5 | cls. top-1 | cls. top-5 | Links |
| - | - | - | - | - | - |
| Pose2Sign (70p face) | 24.41 | 47.59 | 9.74 | 25.99 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_pose2sign_m8_l24_kws8_ps_face/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_pose2sign_m8_l24_kws8_ps_face/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_pose2sign_m8_l24_kws8_ps_face/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_pose2sign_m8_l24_kws8_ps_face/log.json) |
| Pose2Sign (60p body,hands) | 40.47 | 59.45 | 20.24 | 39.27 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_pose2sign_m8_l24_kws8_ps_bodyhands/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_pose2sign_m8_l24_kws8_ps_bodyhands/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_pose2sign_m8_l24_kws8_ps_bodyhands/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_pose2sign_m8_l24_kws8_ps_bodyhands/log.json) |
| Pose2Sign (130p all) | 49.66 | 68.02 | 29.91 | 49.21 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_pose2sign_m8_l24_kws8_ps_allpoints/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_pose2sign_m8_l24_kws8_ps_allpoints/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_pose2sign_m8_l24_kws8_ps_allpoints/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_pose2sign_m8_l24_kws8_ps_allpoints/log.json) |
| I3D (face-crop) | 42.23 | 69.70 | 21.66 | 50.51 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics_ps_facecrop/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics_ps_facecrop/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics_ps_facecrop/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics_ps_facecrop/log.json) |
| I3D (mouth-masked) | 46.75 | 66.34 | 25.85 | 48.02 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics_ps_mouthmasked/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics_ps_mouthmasked/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics_ps_mouthmasked/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics_ps_mouthmasked/log.json) |
| I3D (full-frame) | 65.57 | 81.33 | 44.90 | 64.91 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics_ps_fullframe/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics_ps_fullframe/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics_ps_fullframe/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics_ps_fullframe/log.json) |

* Effect of pretraining

| Model | ins. top-1 | ins. top-5 | cls. top-1 | cls. top-5 | Links |
| - | - | - | - | - | - |
| Random init. | 39.80 | 61.01 | 15.76 | 29.87 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_scratch/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_scratch/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_scratch/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_scratch/log.json) |
| Gesture recognition | 46.93 | 65.95 | 19.59 | 36.44 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pjester/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pjester/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pjester/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pjester/log.json) |
| Sign recognition | 69.90 | 83.45 | 44.97 | 62.73 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pwlasl/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pwlasl/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pwlasl/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pwlasl/log.json) |
| Action recognition | 69.00 | 83.79 | 45.86 | 64.42 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/log.json) |
| Video pose distillation | 70.38 | 84.50 | 46.24 | 65.31 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_ppose/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_ppose/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_ppose/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_ppose/log.json) |

* The effect of the temporal window for KWS

| Model | ins. top-1 | ins. top-5 | cls. top-1 | cls. top-5 | Links |
| - | - | - | - | - | - |
| 1 sec | 60.10 | 75.42 | 36.62 | 53.83 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws1_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws1_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws1_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws1_pkinetics/log.json) |
| 2 sec | 64.91 | 80.98 | 40.29 | 59.63 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws2_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws2_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws2_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws2_pkinetics/log.json) |
| 4 sec | 68.09 | 82.79 | 45.35 | 63.64 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws4_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws4_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws4_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws4_pkinetics/log.json) |
| 8 sec | 69.00 | 83.79 | 45.86 | 64.42 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/log.json) |
| 16 sec | 65.91 | 81.84 | 39.51 | 59.03 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws16_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws16_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws16_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws16_pkinetics/log.json) |

* The effect of the number of frames before the mouthing peak

| Model | ins. top-1 | ins. top-5 | cls. top-1 | cls. top-5 | Links |
| - | - | - | - | - | - |
| 16 frames | 59.53 | 77.08 | 36.16 | 58.43 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l16_kws8_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l16_kws8_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l16_kws8_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l16_kws8_pkinetics/log.json) |
| 20 frames | 71.71 | 85.73 | 49.64 | 69.23 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l20_kws8_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l20_kws8_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l20_kws8_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l20_kws8_pkinetics/log.json) |
| 24 frames | 69.00 | 83.79 | 45.86 | 64.42 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m8_l24_kws8_pkinetics/log.json) |

### Experiments on Transfer
* WLASL dataset (isolated) - 64 frames input

| Model | ins. top-1 | ins. top-5 | cls. top-1 | cls. top-5 | Links |
| - | - | - | - | - | - |
| Kinetics pretraining | 40.85 | 74.10 | 39.06 | 73.33 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/wlasl_i3d_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/wlasl_i3d_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/wlasl_i3d_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/wlasl_i3d_pkinetics/log.json) |
| BSL-1K pretraining | 46.82 | 79.36 | 44.72 | 78.47 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/wlasl_i3d_pbsl1k/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/wlasl_i3d_pbsl1k/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/wlasl_i3d_pbsl1k/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/wlasl_i3d_pbsl1k/log.json) |

* MSASL dataset (isolated) - 64 frames input

| Model | ins. top-1 | ins. top-5 | cls. top-1 | cls. top-5 | Links |
| - | - | - | - | - | - |
| Kinetics pretraining | 60.45 | 82.05 | 57.17 | 80.02 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/msasl_i3d_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/msasl_i3d_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/msasl_i3d_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/msasl_i3d_pkinetics/log.json) |
| BSL-1K pretraining | 64.71 | 85.59 | 61.55 | 84.43 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/msasl_i3d_pbsl1k/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/msasl_i3d_pbsl1k/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/msasl_i3d_pbsl1k/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/msasl_i3d_pbsl1k/log.json) |

* Phoenix2014T dataset (co-articulated) - 16 frames input

| Model | wer | del_rate | ins_rate | sub_rate | Links |
| - | - | - | - | - | - |
| Kinetics pretraining | 45.07 | 22.05 | 6.52 | 16.50 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/phoenix2014t_i3d_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/phoenix2014t_i3d_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/phoenix2014t_i3d_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/phoenix2014t_i3d_pkinetics/log.json) |
| BSL-1K pretraining | 39.49 | 22.54 | 5.03 | 11.92 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/phoenix2014t_i3d_pbsl1k/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/phoenix2014t_i3d_pbsl1k/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/phoenix2014t_i3d_pbsl1k/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/phoenix2014t_i3d_pbsl1k/log.json) |

* BSL-Corpus dataset subset (co-articulated) - 16 frames input

| Model | ins. top-1 | ins. top-5 | cls. top-1 | cls. top-5 | Links |
| - | - | - | - | - | - |
| Kinetics pretraining | 12.79 | 23.11 | 7.76 | 15.76 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bslcp_i3d_pkinetics/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bslcp_i3d_pkinetics/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bslcp_i3d_pkinetics/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bslcp_i3d_pkinetics/log.json) |
| BSL-1K pretraining | 24.35 | 39.14 | 16.00 | 28.54 | [run](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bslcp_i3d_pbsl1k/run.sh), [args](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bslcp_i3d_pbsl1k/opt.pkl), [model](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bslcp_i3d_pbsl1k/model.pth.tar), [logs](https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bslcp_i3d_pbsl1k/log.json) |


### Note on BSL-1K data release
We are in the process of finalising legal confirmation from our broadcasting partners before we release data.

### Limitations

We would like to emphasise that this research represents a working progress towards achieving automatic sign language recognition, and as such, has a number of limitations that we are aware of (and likely many that we are not aware of).  Key limitations include:
* The data collected with our technique is *long-tailed* (this can be seen in Fig. 2 of our paper, referenced below).  This reflects the nature of how signs are used in reality, but it also makes it challenging to train existing vision models (which prefer balanced data).
* All data collected here is *interpreted*.  Interpreted data differs from conversations between native signers (see e.g. [this paper](https://www.microsoft.com/en-us/research/publication/sign-language-recognition-generation-and-translation-an-interdisciplinary-perspective/) for a discussion on this point).
* Our approach naturally biases the annotated data towards mouthings (signs that are not frequently
mouthed, or signers who do not mouth, are less represented).

### Citation
If you use this code, please cite the following:

```
@INPROCEEDINGS{albanie20_bsl1k,
  title     = {{BSL-1K}: {S}caling up co-articulated sign language recognition using mouthing cues},
  author    = {Albanie, Samuel and Varol, G{\"u}l and Momeni, Liliane and Afouras, Triantafyllos and Chung, Joon Son and Fox, Neil and Zisserman, Andrew},
  booktitle = {ECCV},
  year      = {2020}
}
```
