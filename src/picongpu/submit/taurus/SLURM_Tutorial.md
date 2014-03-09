SLURM examples
==============

- `tbg -s sbatch -c submit/0008gpus.cfg -t submit/taurus/k20x_profile.tpl $SCRATCH/runs/test123`

- interactive job:
  - `salloc --time=1:00:00 --gres=gpu:2 --nodes=1 --ntasks-per-node=2 --cpus-per-task=8 --partition gpu-interactive`
  - e.g. `srun "hostname"`

- details for my jobs:
  - `scontrol -d show job 12345`
  - ``` squeue -u `whoami` -l```

- details for queues:
  - `squeue -p queueName -l` (list full queue)
  - `squeue -p queueName --start` (show start times for pending jobs)
  - `squeue -p queueName -l -t R` (only show running jobs in queue)
  - `sview` (alternative on taurus: `module load llview` and `llview`)
  - `scontrol show partition queueName`

- Communicate with job:
  - `scancel 12345` abort job
  - `scancel -s Number 12345` send signal or signal name to job
  - `scontrol update timelimit=4:00:00 jobid=12345`
