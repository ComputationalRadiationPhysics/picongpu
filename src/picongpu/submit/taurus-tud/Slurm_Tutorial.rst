Slurm examples
==============

Job Submission
""""""""""""""

PIConGPU job submission on the *Taurus* cluster at *TU Dresden*:

* `tbg -s sbatch -c submit/0008gpus.cfg -t submit/taurus-tud/k80_profile.tpl $SCRATCH/runs/test123`


Job Control
"""""""""""

* interactive job:

  * `salloc --time=1:00:00 --nodes=1 --ntasks-per-node=2 --cpus-per-task=8 --partition gpu-interactive`
  * e.g. `srun "hostname"`
  * GPU allocation on taurus requires an additional flag, e.g. for two GPUs `--gres=gpu:2`

* details for my jobs:

  * `scontrol -d show job 12345`
  * ``` squeue -u `whoami` -l```

* details for queues:

  * `squeue -p queueName -l` (list full queue)
  * `squeue -p queueName --start` (show start times for pending jobs)
  * `squeue -p queueName -l -t R` (only show running jobs in queue)
  * `sinfo -p queueName` (show online/offline nodes in queue)
  * `sview` (alternative on taurus: `module load llview` and `llview`)
  * `scontrol show partition queueName`

* communicate with job:

  * `scancel 12345` abort job
  * `scancel -s Number 12345` send signal or signal name to job
  * `scontrol update timelimit=4:00:00 jobid=12345` change the walltime of the job
  * `scontrol update jobid=12345 dependency=afterany:54321` only start the job after job with id `54321` has finished
  * `scontrol hold jobid=12345` prevent the job from starting
  * `scontrol unhold jobid=12345` release the job to be eligible for run (after it was set on hold)
