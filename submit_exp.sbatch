#!/bin/bash

### ALIEN PARTITION
#SBATCH --job-name=MIRT_WITH_FACTOR_ANALYSIS_ALL_SCENARIOS
#SBATCH  --output=../logs/%x-%j.out 
#SBATCH  --error=../logs/%x-%j.err 
#SBATCH  --mem=32G
#SBATCH  -c 10
#SBATCH  --qos=alien 
#SBATCH  --gres=gpu:1
#SBATCH  -p alien 
#SBATCH  --exclude=node0[43,44] 
#SBATCH  --time=24:00:00 

module load Miniconda3
source /home/lweber/anaconda3/etc/profile.d/conda.sh
conda activate irt2

python ./mirt_helm.py

exit 0

