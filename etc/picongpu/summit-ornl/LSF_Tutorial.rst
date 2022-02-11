LSF examples
============

Job Submission
''''''''''''''

PIConGPU job submission on the *Summit* cluster at *Oak Ridge National Lab*:

* ``tbg -s bsub -c etc/picongpu/0008gpus.cfg -t etc/picongpu/summit-ornl/gpu.tpl $PROJWORK/$proj/test-001``


Job Control
'''''''''''

* interactive job:

  * ``bsub -P $proj -W 2:00 -nnodes 1 -Is /bin/bash``

* `details for my jobs <https://docs.olcf.ornl.gov/systems/summit_user_guide.html#monitoring-jobs>`_:

  * ``bjobs 12345`` all details for job with <job id> ``12345``
  * ``bjobs [-l]`` all jobs under my user name
  * ``jobstat -u $(whoami)`` job eligibility
  * ``bjdepinfo 12345`` job dependencies on other jobs

* details for queues:

  * ``bqueues`` list queues

* communicate with job:

  * ``bkill <job id>`` abort job
  * ``bpeek [-f] <job id>`` peek into ``stdout``/``stderr`` of a job
  * ``bkill -s <signal number> <job id>`` send signal or signal name to job
  * ``bchkpnt`` and ``brestart`` checkpoint and restart job (untested/unimplemented)
  * ``bmod -W 1:30 12345`` change the walltime of a job (currently not allowed)
  * ``bstop <job id>`` prevent the job from starting
  * ``bresume <job id>`` release the job to be eligible for run (after it was set on hold)


References
''''''''''

* https://docs.olcf.ornl.gov/systems/summit_user_guide.html#running-jobs
* https://www.ibm.com/docs/en/spectrum-lsf
