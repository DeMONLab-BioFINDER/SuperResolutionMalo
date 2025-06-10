#!/bin/bash -l


dic_file="params.json"
raw_path=`jq -r '.path_inference' "$dic_file"`

cd $raw_path
echo $raw_path
mkdir processed/infered

: <<'END'
#If used on 3T images that do not have an associated 7T scan.
path_ref=$1
path_ref_im=$2
for file in 3T/*.nii.gz; do
    if [ -f "$file" ]; then
	    path_out="${file%.nii.gz}_registered.nii.gz"
	    python ../../func/inference/fake_registration.py reg $file $path_out $path_ref $path_ref_im
    fi
done

END


cd ../..

#Infering
python func/inference/inferer3T.py InfSubj "results/trial1/my_unet.pt" "results/trial1/params.txt" "_synthetic7T"

#comparing 
python func/inference/compare.py

echo "done"
