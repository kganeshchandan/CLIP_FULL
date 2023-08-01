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
#SBATCH --output=/home2/kanakala.ganesh/CLIP_PART_1/outputs/first_spec_recon_next_molencoder_next_full.txt

#wget http://www.bindingmoad.org/files/biou/every_part_a.zip
#wget http://www.bindingmoad.org/files/biou/every_part_b.zip

#python sample_run.py

cd ..

python run.py configs/standard/temp_unit_norm.yaml
# python run.py configs/standard/unit_nt_xent.yaml
# python run.py configs/standard/recon_mol_latents.yaml
# python run.py configs/standard/minmax_norm.yaml
# python run.py configs/standard/unit_norm.yaml
