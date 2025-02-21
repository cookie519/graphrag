#!/bin/bash

#SBATCH --job-name=graph     # create a short name for your job

#SBATCH --nodes=1                # node count

#SBATCH --ntasks=1               # total number of tasks across all nodes

#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)

#SBATCH --mem-per-cpu=5G         # memory per cpu-core (4G is default)

#SBATCH --gres=gpu:1            # number of gpus per node

##SBATCH --array=0-9              # job array with index values 0, 1, 2,...

#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)

##SBATCH -p gpu-ee
##SBATCH --partition=pli
##SBATCH --account=llmkg


#SBATCH --mail-type=begin        # send email when job begins

#SBATCH --mail-type=end          # send email when job ends

#SBATCH --mail-type=fail         # send email if job fails

#SBATCH --mail-user=jx0800@princeton.edu
#SBATCH --output=slurm_logs/%A_%a


module purge
module load anaconda3/2024.6
conda activate graphrag
cd /home/jx0800/graphrag

# ollama serve &

# graphrag index --root /scratch/gpfs/jx0800/data/graphrag
#graphrag query --root /scratch/gpfs/jx0800/data/graphrag --method local --query "What's the difference between CNN and GNN? Concluse the pros and cons."
#python test.py

#!/bin/bash

# Start Ollama
# ollama serve &

# graphrag index --root /scratch/gpfs/jx0800/data/graphrag

vllm serve /scratch/gpfs/jx0800/models/Llama-3.2-3B-Instruct --port 11434 & 

# Wait until port 11434 is available
while ! nc -z localhost 11434; do 
  sleep 2
done

python test.py

 
