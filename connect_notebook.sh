#!/bin/bash
# Parameters
#SBATCH --mem=64G
#SBATCH --error=../logs/$jobname%j_0_log.err
#SBATCH --job-name=jupyter
#SBATCH --output=../logs/$jobname%j_0_log.out
#SBATCH --time=12:00:00
#SBATCH --partition=high

cd $HOME/project_fast_eval/efficbench/


$HOME/anaconda3/envs/commentariesAL/bin/jupyter-notebook  --ip $(ip addr show eno1 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1) --no-browser

#$HOME/anaconda3/envs/commentariesAL/bin/jupyter-notebook --ip $(ip addr show eno1 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1) --no-browser
#
