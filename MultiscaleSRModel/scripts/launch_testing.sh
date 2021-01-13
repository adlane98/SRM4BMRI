#!/bin/sh

#SBATCH --job-name=SRReCNN-test
#SBATCH --output=/projets/srm4bmri/outputs/SRReCNN-test.out
#SBATCH --error=/projets/srm4bmri/outputs/SRReCNN-test.err

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

module purge
module load singularity/3.0.3
path_model=/projets/srm4bmri/srrecnn/metadata/new_model

srun singularity exec ${container} ${python} ${script} --test --model ${path_model} --testinput /projets/srm4bmri/originals/marmoset_small/ --output /projets/srm4bmri/srrecnn/outputs/ --downsample
