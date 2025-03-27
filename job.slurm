#!/bin/bash

#SBATCH --job-name=validate     # create a short name for your job

#SBATCH --nodes=1                # node count

#SBATCH --ntasks=1               # total number of tasks across all nodes

#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)

#SBATCH --mem-per-cpu=20G         # memory per cpu-core (4G is default)

#SBATCH --gres=gpu:2            # number of gpus per node

#SBATCH --gpu-mps            # enable GPU Multi-Process Service

##SBATCH --constraint=gpu80

#SBATCH --array=0-8              # job array with index values 0, 1, 2,...

#SBATCH --time=05:00:00          # total run time limit (HH:MM:SS)

##SBATCH -p gpu-ee
#SBATCH --partition=pli
#SBATCH --account=llmkg


#SBATCH --mail-type=begin        # send email when job begins

#SBATCH --mail-type=end          # send email when job ends

#SBATCH --mail-type=fail         # send email if job fails

#SBATCH --mail-user=jx0800@princeton.edu
#SBATCH --output=slurm_logs/%A_%a


module purge
module load anaconda3/2024.6
conda activate graphrag
cd /home/jx0800/graphrag

python validate_triples.py
#python test_max_batch_size.py

# ollama serve &

# graphrag index --root /scratch/gpfs/jx0800/data/graphrag
#graphrag query --root /scratch/gpfs/jx0800/data/graphrag --method local --query "What's the difference between CNN and GNN? Concluse the pros and cons."
#python test.py

#!/bin/bash

# Start Ollama
# ollama serve &

# graphrag index --root /scratch/gpfs/jx0800/data/graphrag

#vllm serve /projects/JHA/jx0800/models/Llama-3.3-70B-Instruct --port 11434 --quantization fp8 --device cuda --tensor-parallel-size 4 --override-generation-config '{"temperature":0.6, "top_p":0.95, "top_k":100, "max_tokens":2048}' #/scratch/gpfs/jx0800/models/Llama-3.2-1B-Instruct
#vllm serve /scratch/gpfs/jx0800/models/Llama-3.2-1B-Instruct --port 11434 --override-generation-config '{"temperature":0.6, "top_p":0.95, "top_k":100, "max_tokens":2048}' #/scratch/gpfs/jx0800/models/Llama-3.2-1B-Instruct


# ollama success
# ollama serve &

# Wait until port 11434 is available
# while ! nc -z localhost 11434; do 
#   sleep 1
# done

# graphrag query --root /scratch/gpfs/jx0800/data/graphrag --method local --query "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma: A. Hyperplasia, B. Hyperophy, C. Atrophy, D. Dyplasia. Please select the correct answer from A, B, C, D."

 
