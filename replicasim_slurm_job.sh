#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000
#SBATCH --time=7-00:00:00
#SBATCH --error="RS_test_%j.err.out"
#SBATCH --output="RS_test_%j.out"
#SBATCH --account=<youraccountname>
#SBATCH --job-name="RS_test"

module load julia

julia --project="." -e "using Pkg; Pkg.instantiate()"
julia --project="." -e "using Pkg; Pkg.status()"


julia --project="." ./code/replica_sim.jl
