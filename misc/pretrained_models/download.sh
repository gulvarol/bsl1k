# create output location
output_dir="misc/pretrained_models"
mkdir -p "${output_dir}"

# BSL-1K (equivalent to https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/experiments/bsl1k_i3d_m5_l20_kws8_ppose/model.pth.tar)
wget https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/pretrained_models/bsl1k.pth.tar -P "${output_dir}"

# BSL-1K mouth-masked pose-pretrained
wget https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/pretrained_models/bsl1k_mouth_masked_ppose.pth.tar -P "${output_dir}"

# Video pose distillation
wget https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/pretrained_models/video_pose_distillation.pth.tar -P "${output_dir}"

# Kinetics
wget https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/pretrained_models/kinetics.pth -P "${output_dir}"

# Jester
wget https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/pretrained_models/jester.pth -P "${output_dir}"

# WLASL 16f
wget https://www.robots.ox.ac.uk/~vgg/research/bsl1k/data/pretrained_models/wlasl16.pth.tar -P "${output_dir}"

