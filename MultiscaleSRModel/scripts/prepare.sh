#!/bin/sh

#SBATCH --job-name=SRReCNN-prepare
#SBATCH --output=/projets/srm4bmri/outputs/SRReCNN-prepare.out
#SBATCH --error=/projets/srm4bmri/outputs/SRReCNN-prepare.err

#SBATCH --mail-type=END   
#SBATCH --mail-user=ladjal.adlane@gmail.com

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --partition=24CPUNodes
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA11/tf2-NGC-20-06-py3.sif
python=$HOME/envs/srrecnn2/bin/python
script=$HOME/SRReCNN/srrecnn.py

module purge
module load singularity/3.0.3

srun singularity exec ${container} ${python} ${script} --prepare --mri /projets/srm4bmri/originals/marmoset_small/ --samples 20
