SLURM examples
==============

- `tbg -s sbatch -c submit/0008gpus.cfg -t submit/pizdaint-cscs/normal_profile.tpl $SCRATCH/runs/test123`

- interactive job:
  - `salloc --time=1:00:00 --nodes=1 --ntasks-per-node=16 --ntasks-per-core=2 --partition normal`
  - `--ntasks-per-core=2` activate intel hyper threading
  - e.g. `aprun "hostname"`


- details for my jobs:
  - `scontrol -d show job 12345`
  - `squeue -u `whoami` -l`

- details for queues:
  - `squeue -p queueName -l` (list full queue)
  - `squeue -p queueName --start` (show start times for pending jobs)
  - `squeue -p queueName -l -t R` (only show running jobs in queue)
  - `sinfo -p queueName` (show online/offline nodes in queue)
  - `scontrol show partition queueName`

- Communicate with job:
  - `scancel 12345` abort job
  - `scancel -s Number 12345` send signal or signal name to job
  - `scontrol update timelimit=4:00:00 jobid=12345`
