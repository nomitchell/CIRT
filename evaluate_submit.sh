#!/bin/bash
#SBATCH --account=def-yymao
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:4
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=72-00:00
#SBATCH --output=HPO-cifar-%A-%a.out
#SBATCH --mail-user=nmitc082@uottawa.ca
#SBATCH --mail-type=ALL

module load python/<3.12> # Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision --no-index
pip install optuna torchattacks tqdm
pip install git+https://github.com/RobustBench/robustbench.git

# $SLURM_ARRAY_TASK_ID


#python evaluate.py  --dataset='imagenet' --epsilon=1 --attack_mode='aa' --def_mode='pRD' --n_clusters=4 --patch_size=1
#python evaluate.py  --dataset='imagenet' --epsilon=1 --attack_mode='aa' --def_mode='swRD' --n_clusters=4 --patch_size=1

python hpo_cirt.py













