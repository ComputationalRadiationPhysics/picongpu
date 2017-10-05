Slurm examples
==============

Job Submission
''''''''''''''

PIConGPU job submission on the *Taurus* cluster at *TU Dresden*:

* ``tbg -s sbatch -c etc/picongpu/0008gpus.cfg -t etc/picongpu/taurus-tud/k80.tpl $SCRATCH/runs/test-001``


Job Control
'''''''''''

* interactive job:

  * ``salloc --time=1:00:00 --nodes=1 --ntasks-per-node=2 --cpus-per-task=8 --partition gpu-interactive``
  * e.g. ``srun "hostname"``
  * GPU allocation on taurus requires an additional flag, e.g. for two GPUs ``--gres=gpu:2``

* details for my jobs:

  * ``scontrol -d show job 12345`` all details for job with <job id> ``12345``
  * ``squeue -u $(whoami) -l`` all jobs under my user name

* details for queues:

  * ``squeue -p queueName -l`` list full queue
  * ``squeue -p queueName --start`` (show start times for pending jobs)
  * ``squeue -p queueName -l -t R`` (only show running jobs in queue)
  * ``sinfo -p queueName`` (show online/offline nodes in queue)
  * ``sview`` (alternative on taurus: ``module load llview`` and ``llview``)
  * ``scontrol show partition queueName``

* communicate with job:

  * ``scancel <job id>`` abort job
  * ``scancel -s <signal number> <job id>`` send signal or signal name to job
  * ``scontrol update timelimit=4:00:00 jobid=12345`` change the walltime of a job
  * ``scontrol update jobid=12345 dependency=afterany:54321`` only start job ``12345`` after job with id ``54321`` has finished
  * ``scontrol hold <job id>`` prevent the job from starting
  * ``scontrol release <job id>`` release the job to be eligible for run (after it was set on hold)
