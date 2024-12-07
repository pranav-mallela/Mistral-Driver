#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=rob535project
#SBATCH --account=na565s001f24_class
#SBATCH --partition=spgpu,gpu_mig40
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=out.txt

eval "$(conda shell.bash hook)"
conda activate ROB535

python3 driving.py