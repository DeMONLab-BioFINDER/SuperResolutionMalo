#!/bin/bash -l

raw_path=$1

echo $raw_path


mkdir $raw_path/../processed
mkdir $raw_path/../npy_files
mkdir $raw_path/../infered

#rm -r $raw_path/../tpt
mkdir $raw_path/../tpt
mkdir $raw_path/../tpt/3T
mkdir $raw_path/../tpt/7T
mkdir $raw_path/../processed/3T
mkdir $raw_path/../processed/7T

mkdir $raw_path/../infered/3T
mkdir $raw_path/../infered/7T

for file in $raw_path/*T/*/*.nii.gz;do
    echo $file
	file2=${file/raw/tpt}
	output_file="${file2%.nii.gz}_brain.nii.gz"
	output_mask="${file2%.nii.gz}_mask.nii.gz"
	mkdir $(dirname $file2)

    if [ -f "$file" ] && [ ! -f "$output_file" ]; then
		mri_synthstrip -i $file -o $output_file -m $output_mask --no-csf
    fi
done
echo "done stripping"


for file in $raw_path/*T/*/*.nii.gz;do
	file2=${file/raw/tpt}
	output_file="${file2%.nii.gz}_corrected.nii.gz"
	mkdir $(dirname $file2)

    if [ -f "$file" ] && [ ! -f "$output_file" ]; then
		file_mask="${file2%.nii.gz}_mask.nii.gz"
	    python func/processing/BiasCorrection.py correc $file $file_mask $output_file
		echo "done"
    fi
done

echo "Done correcting"


processed_path=$raw_path/../processed

for file in $raw_path/../tpt/*T/*/*corrected.nii.gz;do
	file2=${file/tpt/processed}
	output_file="${file2%.nii.gz}_brain.nii.gz"
	mkdir $(dirname $output_file)
    if [ -f "$file" ] && [ ! -f "$output_file" ]; then
		mri_synthstrip -i $file -o $output_file --no-csf
    fi
done
#rm -r $raw_path/../tpt

echo "Done stripping, registering"

for file in $processed_path/3T/*/*_corrected_brain.nii.gz; do
	path_out="${file%.nii.gz}_registered.nii.gz"
    if [ -f "$file" ] && [ ! -f "$path_out" ]; then
	    python func/processing/registration.py reg $file $path_out
    fi
done


echo "done preprocessing"