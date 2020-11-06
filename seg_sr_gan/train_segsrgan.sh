#!/bin/sh

#SBATCH --job-name=SegSRGAN-1st-Train
#SBATCH --output=SegSRGAN-1st-Train.out
#SBATCH --error=SegSRGAN-1st-Train.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA11/tf2-NGC-20-06-py3.sif
python=/users/aladjal/envs/segsrgan/bin/python
script=/users/aladjal/SegSRGAN/train.py
data=data_marmoset.csv

module purge
module load singularity/3.0.3
srun singularity exec ${container} ${python} ${script} --csv ${data}
