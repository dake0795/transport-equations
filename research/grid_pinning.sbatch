#!/bin/bash
#SBATCH -J grid_pinning
#SBATCH -A ukaea-ap001-cpu
#SBATCH -p ukaea-icl
#SBATCH --qos=t2-lowu-cpu1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o grid_pinning_%j.log
cd /rds/project/rds-aSo1XX0UOlw/ir-kenn3/TRANSPORT_EQUATIONS/transport-equations/research
export OMP_NUM_THREADS=4
python3 -u exp_grid_pinning.py 50 100 200 400 > grid_pinning_resolved.out
