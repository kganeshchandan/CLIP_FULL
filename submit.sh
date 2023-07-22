#!/bin/bash

#SBATCH --job-name=MultiGPU-TestJob
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -c 40
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-user=kanakala.ganesh@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=outputs/op_file_multigpu_smiles_correct_total.txt

#wget http://www.bindingmoad.org/files/biou/every_part_a.zip
#wget http://www.bindingmoad.org/files/biou/every_part_b.zip

#python sample_run.py


python run_smiles.py
