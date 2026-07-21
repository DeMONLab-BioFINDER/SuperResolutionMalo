#!/bin/bash -l

dic_file="params.json"

#Image preprocessing
path=`jq -r '.path_data' "$dic_file"`
infere_mode=`jq -r '.infere_mode' "$dic_file"`
raw_path=${path}raw
source script/processing_subpipeline.sh $raw_path

echo "Getting the limits"
python func/processing/get_lims.py lims "path_patient_info" "path_data"


echo "Saving as npy files"
python func/images_to_npy/pickling.py "path_data"

echo "done"