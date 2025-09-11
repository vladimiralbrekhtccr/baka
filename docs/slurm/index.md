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


## How to take recources from the node:

You can name Name this file as salloc_res.sh 

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

After you received the recources you can do:

```bash
ssh node006
```

And work as you were working usually on H100.

