#!/bin/bash -l


dic_file="params.json"
raw_path=`jq -r '.path_data' "$dic_file"`
model_path=`jq -r '.path_inference_model' "$dic_file"`
params_path=`jq -r '.path_inference_model_params' "$dic_file"`
infere_mode=`jq -r '.infere_mode' "$dic_file"`

echo $raw_path
mkdir $raw_path/processed/infered

echo "Infering"
python func/inference/inferer3T.py InfSubj $model_path $params_path "_synthetic7T"

if [ "$infere_mode" != "True" ]; then
	echo "Comparing"
	python func/inference/compare.py
fi
echo "done"