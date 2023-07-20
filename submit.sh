#!/bin/bash

#SBATCH --job-name=TestJob
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -c 30
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=kanakala.ganesh@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=op_file3.txt

#wget http://www.bindingmoad.org/files/biou/every_part_a.zip
#wget http://www.bindingmoad.org/files/biou/every_part_b.zip

#python sample_run.py

python run.py