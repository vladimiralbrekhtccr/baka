# Slurm info


## Comands to check status

How to check running jobs

```bash
squeue 
```

How to check avaliable recources

```bash
sinfo 
```

sHow to stop the job

```bash
scancel <JOB_ID>  # example `scancel 35512`
```


## Salloc How to take resources on the node (interactvie mode):

You can name this file as salloc_res.sh 

and run with `bash salloc_res.sh`

```bash
SESSION_NAME="2GPUs_int_yolo_trian"

if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux new-session -d -s $SESSION_NAME

    tmux send-keys -t $SESSION_NAME "salloc \
      --partition=defq \
      --time=167:00:00 \
      --nodelist=node006 \
      --nodes=1 \
      --mem=1 \
      --job-name='2GPUs_int_yolo_trian'" C-m
fi

tmux attach-session -t $SESSION_NAME
```

After you received the resources you can do:

```bash
ssh node006
```

And work as you were working usually on H100.


## Only send 1 request using

From Head_node run `sbatch <your_bash_script.sh>` in terminal it will create a job for the node.

`your_bash_script.sh`
```bash
#!/bin/bash
#SBATCH --job-name=qwen-train                           # job name
#SBATCH --nodes=2                                       # amount of nodes to take
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=64                              # amount of cpus to take
#SBATCH --nodelist=node00[1-2]                          # which nodes you want to take
#SBATCH --gres=gpu:8                                    # amount of gpus to take
#SBATCH --time=168:00:00                                # max time until job
#SBATCH --mem=1000000M                                  # amount of RAM to take
#SBATCH --partition=defq
#SBATCH --output=logs_internvl/slurm-%N.%j.out          # where out logs will be writen
#SBATCH --error=logs_internvl/slurm-%N.%j.err           # where error logs will be writen

# Conda initialization
eval "$(conda shell.bash hook)"
conda activate axolotl-upd

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1

# # Setting master address and port
# MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

srun torchrun \
    --nnodes 2 \
    --nproc_per_node 8 \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_id $SLURM_JOB_ID \
    -m axolotl.cli.train --config ${CONFIG_PATH}
```