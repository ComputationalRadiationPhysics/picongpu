.. _checkpoint:

Checkpoints
===========

The checkpoint plugin writes the *complete* state of the simulation
(fields, particles, solver state) to disk on a schedule,
so that a run can be resumed from the last written checkpoint
instead of from the beginning.
This is the standard tool for

* long production runs that must survive node failures, and
* restarting a run with a modified input
  (e.g. extended run length or a different laser)
  without recomputing the past.

.. literalinclude:: ../snippets/selected_topics/checkpoint.py
   :language: python
   :start-after: checkpoint = Checkpoint
   :end-before: # A follow-up run

At least one of the two scheduling parameters is required:

* ``period``:
  a :ref:`time-step specification <time-steps>` --
  checkpoints at these steps; or
* ``timePeriod``:
  a time interval in **minutes** of wall-clock time
  (checkpoints when the elapsed time exceeds the interval).

Other parameters:

* ``directory``:
  the directory *inside the output* where checkpoints are written
  (default ``"checkpoints"``).
* ``file``:
  the file-set prefix for the checkpoint files
  (the plugin appends the step number).
* ``openPMD``:
  a dictionary of openPMD-specific options
  (``ext``, ``infix``, ``backendConfig``, ...)
  for the checkpoint files.

Restarting a run
----------------

A new run resumes from a checkpoint by giving it the restart options
instead of a fresh ``period``:

``restart=True``
   Restart from the latest checkpoint
   (or from the specific step given via ``restartStep``;
   the run aborts if the checkpoint does not exist).

``tryRestart=True``
   Restart from the latest available checkpoint,
   or start from scratch if none exists --
   the safe default for a resubmitted run.

``restartStep``
   the specific checkpoint step to restart from
   (default: the latest one).

``restartDirectory`` / ``restartFile``
   where to read the checkpoint from
   (defaults: ``"checkpoints"`` / the value of ``file``).

``restartChunkSize``
   the number of particles processed per kernel call during restart
   (larger is faster, but needs more memory).

``restartLoop``
   how many times the simulation should be restarted after it finishes
   (for automatic continuation runs).

.. note::

   The checkpoint directory is part of the job's output
   (see :ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>`),
   so for a restart the checkpoint files must be available on the
   restart job -- typically by pointing the restart at the finished
   run directory.
