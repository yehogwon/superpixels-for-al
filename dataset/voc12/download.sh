#!/bin/bash

cwd=$(pwd)
timestamp=$(date +%s)
download_path=/tmp/ss-dataset-voc12-$timestamp

mkdir -p $download_path
cd $download_path

wget http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar

tar -xvf VOCtrainval_11-May-2012.tar
# VOCdevkit/VOC2012/ -> 5 directories

mv VOCdevkit/VOC2012/JPEGImages $cwd
mv VOCdevkit/VOC2012/SegmentationClass $cwd
mv VOCdevkit/VOC2012/ImageSets/Segmentation $cwd

cd $cwd

# remove temporary files
rm -rf $download_path

# remove variables
unset cwd
unset timestamp
unset download_path
