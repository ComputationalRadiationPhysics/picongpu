.. _in5min:

PIConGPU in 5 Minutes
=====================

A guide to run, but not understand PIConGPU.
It is aimed at existing and preconfigured high performance computing clusters (“supercomputers”, also: HPC systems).
For more information on the individual steps please refer to the manual.

This guide needs **shell access** (probably via ``ssh``) and **some auxilliary programs** (e.g. ``git``).
Consider getting familiar with the shell and git.
Please also read the tutorial for your local HPC cluster.

We will use the following directories:

- ``~/src/picongpu``: source files from github
- ``~/picongpu.profile``: load the dependencies for your local environment
- ``~/picongpu-projects``: scenarios to simulate
- ``/scratch/mydir``: result data of the simulation runs

Please replace them whenever appropriate.

Get the Source
--------------

Use git to obtain the source and use the current ``dev`` branch and put it into ``~/src/picongpu``::

  mkdir -p ~/src
  git clone -b dev https://github.com/ComputationalRadiationPhysics/picongpu ~/src/picongpu

Setup
-----

You need :ref:`a lot of dependencies <install-dependencies>`.

If you are on a known cluster you can take a shortcut:
Select a profile file (``NAME.profile.example``) for your local HPC cluster from ``etc/picongpu`` and copy it to your home directory::

  cp ~/src/picongpu/etc/picongpu/hemera-hzdr/k80_picongpu.profile.example ~/picongpu.profile

This profile determines which part of the HPC cluster (“partition”, also: “queue”) – and thereby the type of GPUs – you will use.

**Maybe you have to adjust this profile file.**
Use your favorite editor.
If unsure use nano: ``nano ~/picongpu.profile`` (Save with Ctrl+O, exit with Ctrl+X)
Read the file (it’s short) and apply changes if appropriate.

For this guide, add after the last line::

  export SCRATCH=/scratch/mydir

This is the location where runtime data and all results will be stored.
Obviously change the path according to your local setup.
Consult the documentation of your HPC cluster where to save your data.
**On HPC clusters this is probably not your home directory.**

Now activate your profile::

  source ~/picongpu.profile

You will have to repeat the last command every time you want to use PIConGPU.

Create a Scenario
-----------------

Create a directory and copy an example::

  mkdir -p ~/picongpu-projects/tinkering
  pic-create $PIC_EXAMPLES/LaserWakefield ~/picongpu-projects/tinkering/try01
  cd ~/picongpu-projects/tinkering/try01

Usually you would now adjust the files in the newly created directory ``~/picongpu-projects/tinkering/try01`` – for this introduction we will use the parameters as provided.
Also note how the variable ``$PIC_EXAMPLES`` has been provided becauses you executed ``source ~/picongpu.profile`` in the previous step.

Compile and Run
---------------

**Now use a compute node.**
Your profile probably provides a helper command for that::

  getDevice

(You can now run ``hostname`` to see which node you are using.)

Now build (from inside your scenario directory, maybe use ``cd ~/picongpu-projects/tinkering/try01``)::

  pic-build

This will take a while, go grab a coffee.
If this fails, read the manual or ask a colleague.

After a successfull build, run (still on the compute node, from inside your scenario directory, maybe use ``cd ~/picongpu-projects/tinkering/try01``)::

  tbg -s bash -c etc/picongpu/1.cfg -t etc/picongpu/bash/mpiexec.tpl $SCRATCH/tinkering/try01/run01

- ``tbg``: tool provided by PIConGPU
- ``bash``: the “submit system”, e.g. use ``sbatch`` for slurm
- ``etc/picongpu/1.cfg``: runtime options (number of GPUs, etc.)
- ``etc/picongpu/bash/mpiexec.tpl``: options for the chosen submit system
- ``$SCRATCH/tinkering/try01/run01``: not-yet-existing destination for your result files

E.g. for “Hypnos” the invocation could be (invoke from the login node)::

  tbg -s qsub -c etc/picongpu/1.cfg -t etc/picongpu/hypnos-hzdr/k20.tpl $SCRATCH/tinkering/try01/run01

Examine the Results
-------------------

Results are located at ``$SCRATCH/tinkering/try01/run01``.

To view pretty pictures from a linux workstation you can use the following process (execute on your workstation, **not the HPC cluster**):

0. Create a “mount point” (empty directory): ``mkdir -p ~/mnt/scratch``
1. Mount the data directory using sshfs:
   ``sshfs -o default_permissions -o idmap=user -o uid=$(id -u) -o gid=$(id -g) HOST:DATADIR ~/mnt/scratch/``
   Substitute HOST with the hostname (``ssh HOST`` should connect to the HPC cluster) and DATADIR with the full path to your data directory, e.g. ``/scratch/mydir``
2. Browse the directory using a file browser/image viewer
   (e.g. ``gwenview``). Check out ``~/mnt/scratch/tinkering/try01/run01/simOutput/pngElectronsYX/`` for image files.

Further Reading
---------------

You now know the process of using PIConGPU.
Carry on reading the documentation to understand it.
