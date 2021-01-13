#!/bin/sh

#SBATCH --job-name=SRReCNN-train
#SBATCH --output=/projets/srm4bmri/outputs/SRReCNN-train.out
#SBATCH --error=/projets/srm4bmri/outputs/SRReCNN-train.err

#SBATCH --mail-type=END   
#SBATCH --mail-user=ladjal.adlane@gmail.com

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA11/tf2-NGC-20-06-py3.sif
python=$HOME/envs/srrecnn2/bin/python
script=$HOME/SRReCNN/srrecnn.py

hdf5_data="/projets/srm4bmri/srrecnn/hdf5_data/20210110-191043.txt"

module purge
module load singularity/3.0.3

srun singularity exec ${container} ${python} ${script} --train -i ${hdf5_data} -l 10 --numkernel 64 -k 3 -b 64 --epochs 20
