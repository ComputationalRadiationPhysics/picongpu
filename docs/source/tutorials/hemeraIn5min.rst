.. _hemeraIn5min:

PIConGPU in 5 Minutes on Hemera
===============================

A guide to run, but not understand PIConGPU.
It is aimed at users of the high performance computing (HPC) cluster `"Hemera" at the HZDR <https://www.hzdr.de/db/Cms?pOid=12231&pNid=852>`_,
but should be applicable to other HPC clusters with slight adjustments.

This guide needs **shell access** (probably via :command:`ssh`) and :command:`git`.
Consider getting familiar with the shell (*command line*, usually :command:`bash`) and git.
Please also read the tutorial for your local HPC cluster.

.. seealso::
   resources for the command line (bash)
     `a tutorial <http://www.bu.edu/tech/files/2018/05/2018-Summer-Tutorial-Intro-to-Linux.pdf>`_ |
     `another tutorial <https://cscar.research.umich.edu/wp-content/uploads/sites/5/2016/09/Intro-to-Command-Line.pdf>`_ |
     `scripting by examples <https://learnxinyminutes.com/docs/bash/>`_

   resources for git
     `official tutorial <https://git-scm.com/docs/gittutorial>`_ (also available as man page :manpage:`gittutorial(7)`) |
     `w3school tutorial <https://www.w3schools.com/git/default.asp>`_ |
     `brief introduction <https://learnxinyminutes.com/docs/git/>`_ |
     `cheatsheet (by github) <https://training.github.com/downloads/github-git-cheat-sheet.pdf>`_

   Hemera at HZDR
     `official website <https://www.hzdr.de/db/Cms?pOid=12231&pNid=852>`_ |
     `presentation <https://www.hzdr.de/db/Cms?pOid=61949>`_
     internal links:
     `wiki <https://fwcc.pages.hzdr.de/infohub/hpc/hemera.html>`_ |
     `storage layout <https://fwcc.pages.hzdr.de/infohub/hpc/storage.html>`_
  
We will use the following directories:

- :file:`~/src/picongpu`: source files from github
- :file:`~/k80_picongpu.profile`: load the dependencies for your local environment
- :file:`~/picongpu-projects`: scenarios to simulate
- :file:`/bigdata/hplsim/external/alice`: result data of the simulation runs (*scratch* storage)

Please replace them whenever appropriate.

Get the Source
--------------

Use :command:`git` to obtain the source and use the current ``dev`` branch and put it into ``~/src/picongpu``::

  mkdir -p ~/src
  git clone https://github.com/ComputationalRadiationPhysics/picongpu ~/src/picongpu

.. note::
   If you get the error ``git: command not found`` load git by invoking ``module load git`` and try again.
   Attention: the example uses the ``dev`` branch instead of the latest stable release.
   Due to driver changes on hemera the modules configuration of the last release might be outdated.

Setup
-----

You need :ref:`a lot of dependencies <install-dependencies>`.

Luckily, other people already did the work and prepared a *profile* that you can use.
Copy it to your home directory::

  cp ~/src/picongpu/etc/picongpu/hemera-hzdr/k80_picongpu.profile.example ~/k80_picongpu.profile

This profile determines which part of the HPC cluster (*partition*, also: *queue*) – and thereby the compute device(s) (type of CPUs/GPUs) – you will use.
This particular profile will use `NVIDIA Tesla K80 <https://www.nvidia.com/en-gb/data-center/tesla-k80/>`_ GPUs.

You can view the full list of available profiles `on github <https://github.com/ComputationalRadiationPhysics/picongpu/tree/dev/etc/picongpu>`_ (look for :file:`NAME.profile.example`).

For this guide we will add our scratch directory location to this profile.
Edit the profile file using your favorite editor.
If unsure use nano: ``nano ~/k80_picongpu.profile`` (save with :kbd:`Control-o`, exit with :kbd:`Control-x`).
Go to the end of the file and add a new line::

  export SCRATCH=/bigdata/hplsim/external/alice

(Please replace ``alice`` with your username.)

.. note::
    This is the location where runtime data and all results will be stored.
    If you're not on Hemera make sure you select the correct directory:
    Consult the documentation of your HPC cluster where to save your data.
    **On HPC clusters this is probably not your home directory.**

In the profile file you can also supply additional settings, like your email address and notification settings.

Now activate your profile::

  source ~/k80_picongpu.profile

.. warning::
   You will have to repeat this command **every time** you want to use PIConGPU on a new shell, i.e. after logging in.

Now test your new profile::

  echo $SCRATCH

That should print your data directory.
If that works make sure that this directory actually exists by executing::

  mkdir -p $SCRATCH
  ls -lah $SCRATCH

If you see output similar to this one everything worked and you can carry on::

  total 0
  drwxr-xr-x  2 alice    fwt   40 Nov 12 10:09 .
  drwxrwxrwt 17 root     root 400 Nov 12 10:09 ..

Create a Scenario
-----------------

As an example we will use the predefined `LaserWakefield example <https://github.com/ComputationalRadiationPhysics/picongpu/tree/dev/share/picongpu/examples/LaserWakefield>`_.
Create a directory and copy it::

  mkdir -p ~/picongpu-projects/tinkering
  pic-create $PIC_EXAMPLES/LaserWakefield ~/picongpu-projects/tinkering/try01
  cd ~/picongpu-projects/tinkering/try01

Usually you would now adjust the files in the newly created directory ``~/picongpu-projects/tinkering/try01`` – for this introduction we will use the parameters as provided.

.. note::
   The command :command:`pic-create` and the variable ``$PIC_EXAMPLES`` have been provided because you loaded the file :file:`~/k80_picongpu.profile` in the previous step.
   If this fails (printing ``pic-create: command not found``), make sure you load the PIConGPU profile by executing ``source ~/k80_picongpu.profile``.

Compile and Run
---------------

**Now use a compute node.**
Your profile provides a helper command for that::

  getDevice

(You can now run ``hostname`` to see which node you are using.)

Now build the scenario::

  # switch to the scenario directory if you haven't already
  cd ~/picongpu-projects/tinkering/try01
  pic-build

This will take a while, go grab a coffee.
If this fails, read the manual or ask a colleague.

After a successfull build, run (still on the compute node, still inside your scenario directory)::

  tbg -s bash -t $PICSRC/etc/picongpu/bash/mpiexec.tpl -c /etc/picongpu/1.cfg $SCRATCH/tinkering/try01/run01

- :command:`tbg`: tool provided by PIConGPU
- ``bash``: the “submit system”, e.g. use ``sbatch`` for slurm
- ``$PICSRC``: the path to your PIConGPU source code, automatically set when sourcing :file:`k80_picongpu.profile`
- :file:`$PICSRC/etc/picongpu/bash/mpiexec.tpl`: options for the chosen submit system
- :file:`etc/picongpu/1.cfg`: runtime options (number of GPUs, etc.)
- :file:`$SCRATCH/tinkering/try01/run01`: not-yet-existing destination for your result files

.. note::
   Usually you would use the *workload manager* (`SLURM <https://slurm.schedmd.com/>`_ on Hemera) to submit your jobs
   instead of running them interactively like we just did.
   You can try that with::

     # go back to the login node
     exit
     hostname
     # ...should now display hemera4.cluster or hemera5.cluster

     # resubmit your simulation with a new directory:
     tbg -s sbatch -c etc/picongpu/1.cfg -t etc/picongpu/hemera-hzdr/k80.tpl $SCRATCH/tinkering/try01/run02

   This will print a confirmation message (e.g. ``Submitted batch job 3769365``),
   but no output of PIConGPU itself will be printed.
   Using ``squeue -u $USER`` you can view the current status of your job.

   Note that we not only used a different "submit system" ``sbatch``,
   but also changed the template file to :file:`etc/picongpu/hemera-hzdr/k80.tpl`.
   (This template file is directly located in your project directory.`)
   Both profile and template file are built for the same compute device, the NVIDIA Tesla "K80" GPU.
   

Examine the Results
-------------------

Results are located at :file:`$SCRATCH/tinkering/try01/run01`.

To view pretty pictures from a linux workstation you can use the following process (execute on your workstation, **not the HPC cluster**)::

  # Create a “mount point” (empty directory)
  mkdir -p ~/mnt/scratch

  # Mount the data directory using sshfs
  sshfs -o default_permissions -o idmap=user -o uid=$(id -u) -o gid=$(id -g) hemera5:DATADIR ~/mnt/scratch/

Substitute DATADIR with the full path to your data (*scratch*) directory, e.g. :file:`/bigdata/hplsim/external/alice`.

Browse the directory using a file browser/image viewer.
Check out :file:`~/mnt/scratch/tinkering/try01/run01/simOutput/pngElectronsYX/` for image files.

Further Reading
---------------

You now know the process of using PIConGPU.
Carry on reading the documentation to understand it.
