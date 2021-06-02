.. _expert-signals:

PIConGPU SIGNALS
================

Sending signals to PIConGPU allows creating checkpoints during the run and a 
clean shutdown before the simulation arrived the end time step.
Signal support is not available on WINDOWS operating systems.

Triggering a checkpoint with signals is only possible if you enabled a :ref:`checkpointing plugin<usage-plugins-checkpoint>`.

Overview
--------

SIGNALS handled by PIConGPU on POSIX systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- HUP   (1): Triggers USR2. Controlling process/terminal hangup.
- INT   (2): Triggers USR2. This SIGNAL gets triggert while hitting ^C.
- QUIT  (3): Triggers USR2. This is the terminal quit SIGNAL.
- ABRT  (6): Triggers USR2. Can only be called from within the code.
- USR1 (10): Create a checkpoint for the next time step.
- USR2 (12): Finish current loop and exit normally by setting time step ``n_max = n``.
- ALRM (14): Trigger USR1 and USR2.
- TERM (15): Trigger USR1.


Default SIGNALS
^^^^^^^^^^^^^^^

These can not be handled:

- KILL  (9)
- CONT (18)
- STOP (19)


Batch Systems
-------------

Slurm
^^^^^

Documenation: https://slurm.schedmd.com/scancel.html

``scancel --signal=USR1  --batch <jobid>``

IBM LSF
^^^^^^^

Documentation: https://www.ibm.com/docs/hu/spectrum-lsf/10.1.0?topic=job-send-signal
``bkill -s USR1 <jobid>``
  

Reference to SIGNALS
--------------------

LINUX SIGNALS: 
  * https://man7.org/linux/man-pages/man7/signal.7.html
  * http://en.wikipedia.org/wiki/Unix_signal#POSIX_signals
