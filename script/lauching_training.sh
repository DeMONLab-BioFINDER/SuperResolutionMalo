#!/bin/bash -l

#source script/processing_pipeline.bash
i=$1

mkdir -p results/trial$i/images

python func/training/LoadingModel.py $i

echo "done"
