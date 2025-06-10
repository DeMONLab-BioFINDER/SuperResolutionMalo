#!/bin/bash -l

raw_path=$1
cd $raw_path
echo $raw_path


mkdir ../processed
mkdir ../npy_files

rm -r ../tpt
mkdir ../tpt
mkdir ../tpt/3T
mkdir ../tpt/7T


for file in *T/*.nii.gz;do
    if [ -f "$file" ]; then
	    echo "$file" >> ../tpt/input.txt
		
		output_file="../tpt/${file%.nii.gz}_brain.nii.gz"
		output_mask="$../tpt/${file%.nii.gz}_mask.nii.gz"
		echo "$output_file" >> ../tpt/output.txt
		echo "$output_masks" >> ../tpt/masks.txt
    fi
done
mri_synthstrip -i ../tpt/input.txt -o ../tpt/output.txt -m ../tpt/masks.txt --threads 16
echo "done stripping"


for file in 3T/*.nii.gz;do
    if [ -f "$file" ]; then
	    output_file="../processed/${file%.nii.gz}_corrected.nii.gz"
		file_mask="$../tpt/${file%.nii.gz}_mask.nii.gz"
	    python ../../../func/BiasCorrection.py correc3T $file $file_mask $output_file
		echo "done"
    fi
done

for file in 7T/*.nii.gz;do
    if [ -f "$file" ]; then
	    output_file="../processed/${file%.nii.gz}_corrected.nii.gz"
		file_mask="$../tpt/${file%.nii.gz}_mask.nii.gz"
	    python ../../../func/processing/BiasCorrection.py correc7T $file $file_mask $output_file
		echo "done"
    fi
done
echo "Done correcting"
rm -r ../tpt
mkdir ../tpt

cd ../processed

for file in *T/*corrected.nii.gz;do
    if [ -f "$file" ]; then
	    echo "$file" >> ../tpt/input.txt
		
		output_file="../tpt/${file%.nii.gz}_brain.nii.gz"
		echo "$output_file" >> ../processed/output.txt
    fi
done
mri_synthstrip -i ../tpt/input.txt -o ../tpt/output.txt --threads 16
echo "Done stripping"


for file in 3T/*.nii.gz; do
    if [ -f "$file" ]; then
	    path_out="${file%.nii.gz}_registered.nii.gz"
	    python ../../../func/processing/registration.py reg $file $path_out
    fi
done


echo "done preprocessing"