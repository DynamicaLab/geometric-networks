#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --mail-user=antoine.legare.1@ulaval.ca
#SBATCH --mail-type=ALL

nvidia-smi

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch --no-index
pip install numpy --no-index
pip install scipy --no-index
pip install scikit-learn --no-index
pip install tqdm --no-index
pip install matplotlib --no-index
pip install statsmodels --no-index

python script.py
