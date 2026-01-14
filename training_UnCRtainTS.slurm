#!/bin/bash
#SBATCH --job-name=uncrtaints_train     # nom du job
#SBATCH --nodes=1                    # on demande 2 noeuds
#SBATCH --ntasks-per-node=1          # avec 2 tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=36           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00             # temps maximum d'execution demande (HH:MM:SS)
##SBATCH --time=00:30:00             # temps maximum d'execution demande (HH:MM:SS)
##SBATCH --qos=qos_gpu-dev
#SBATCH -e /lustre/fsn1/worksf/projects/rech/tel/uug84ql/cloud_reconstruction/logs/uncrtaints_train-%j.out
#SBATCH -o /lustre/fsn1/worksf/projects/rech/tel/uug84ql/cloud_reconstruction/logs/uncrtaints_train-%j.out
#SBATCH --account=tel@v100

module purge

export SRUN_CPUS_PER_TASK=36
export SBATCH_DEBUG=1
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1
export CLEARML_OFFLINE_MODE=1

module load python

conda deactivate
conda activate cloud_reconstruction

set -x

srun --cpu-bind=none python /lustre/fsn1/worksf/projects/rech/tel/uug84ql/cloud_reconstruction/UnCRtainTS/model/train_reconstruct.py --experiment_name jzay_experiment --res_dir /lustre/fsn1/worksf/projects/rech/tel/uug84ql/cloud_reconstruction/outputs/UnCRtainTS