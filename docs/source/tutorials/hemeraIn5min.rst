.. _hemeraIn5min:

PIConGPU in 5 Minutes on RoSi
=============================

A guide to run, but not understand PIConGPU.
It is aimed at users of the high performance computing (HPC) cluster `"RoSi" at the HZDR <https://www.hzdr.de/db/Cms?pOid=12231&pNid=852>`_,
but should be applicable to other HPC clusters with slight adjustments.

This guide needs **shell access** (probably via :command:`ssh`) and :command:`git` (preinstalled on most systems including RoSi).
The RoSi shell can also be used from a browser (currently there is no support for Firefox) at `<https://rosi.hzdr.de/>`_ (accessible from the intranet).
Consider getting familiar with the shell (*command line*, usually :command:`bash`) and git.
Please also read the tutorial for your local HPC cluster.

.. seealso::
   resources for the command line (bash)
     `a tutorial <http://www.bu.edu/tech/files/2018/05/2018-Summer-Tutorial-Intro-to-Linux.pdf>`_ |
     `another tutorial <https://cscar.research.umich.edu/wp-content/uploads/sites/5/2016/09/Intro-to-Command-Line.pdf>`_ |
     `scripting by examples <https://learnxinyminutes.com/docs/bash/>`_ |
     `bash cheatsheet <https://devhints.io/bash>`_

   resources for git
     `official tutorial <https://git-scm.com/docs/gittutorial>`_ (also available as man page :manpage:`gittutorial(7)`) |
     `w3school tutorial <https://www.w3schools.com/git/default.asp>`_ |
     `brief introduction <https://learnxinyminutes.com/docs/git/>`_ |
     `cheatsheet (by github) <https://training.github.com/downloads/github-git-cheat-sheet.pdf>`_

   RoSi at HZDR
     `official website <https://www.hzdr.de/db/Cms?pOid=12231&pNid=852>`_ |
     introduction workshop videos: `Part 1 <https://www.hzdr.de/db/!MediaPlayer?pNid=no&pUrl=/intern/Hoersaal/2025-09-22-10-00-00_HPC%20Workshop%20-%20Introduction%20to%20RoSI%20-%20Day%201--H.%20Schulz%2C%20M.%20Lokamani%2C%20K.%20Ramakrishna%2C%20V.%20Sudharshnam.mp4>`_
     `Part 2 <https://www.hzdr.de/db/!MediaPlayer?pNid=no&pUrl=/intern/Hoersaal/2025-09-23-10-00-00_HPC%20Workshop%20-%20Introduction%20to%20RoSI%20-%20Day%202--H.%20Schulz%2C%20M.%20Lokamani%2C%20K.%20Ramakrishna%2C%20V.%20Sudharshnam.mp4>`_ |
     internal links:
     `wiki <https://fwcc.pages.hzdr.de/infohub/index.html>`_ |
     `storage layout <https://fwcc.pages.hzdr.de/infohub/hpc/storage.html>`_ |
     `using jupyter notebook on RoSi <https://fwcc.pages.hzdr.de/infohub/hpc/rosi_interactive.html#jupyter-notebooks-on-open-ondemand>`_ |
     `in browser code editor for editing files on RoSi <https://fwcc.pages.hzdr.de/infohub/hpc/rosi_interactive.html#code-server-web-based-vs-code-on-open-ondemand>`_

We will use the following files and directories:

- :file:`~/src/picongpu`: source files from GitHub
- :file:`~/gpu-v100_picongpu.profile`: load the dependencies for your local environment
- :file:`~/picongpu-projects`: scenarios to simulate
- :file:`/bigdata/hplsim/scratch/<username>`: result data of the simulation runs (*scratch* storage).
 External users may have to use  :file:`/bigdata/hplsim/external/<username>` instead.

Please replace them whenever appropriate.
Replace ``<username>`` with your username, you can find it by running the :command:`whoami` command on the cluster.

Get the Source
--------------

Use :command:`git` to obtain the source and use the current ``dev`` branch and put it into ``~/src/picongpu``::

  mkdir -p ~/src
  git clone https://github.com/ComputationalRadiationPhysics/picongpu ~/src/picongpu

.. note::
   If you get the error ``git: command not found`` load git by invoking ``module load git`` and try again.
   Attention: the example uses the ``dev`` branch instead of the latest stable release.

Setup
-----

You need :ref:`a lot of dependencies <install-dependencies>`.

Luckily, other people already did the work and prepared a *profile* that you can use.
Copy it to your home directory::

  cp ~/src/picongpu/etc/picongpu/rosi-hzdr/gpu-v100_picongpu.profile.example ~/gpu-v100_picongpu.profile

This profile determines which part of the HPC cluster (*partition*, also: *queue*) – and thereby the compute device(s) (type of CPUs/GPUs) – you will use.
This particular profile will use `NVIDIA Volta V100 <https://www.nvidia.com/en-us/data-center/v100/>`_ GPUs.

You can view the full list of available profiles `on GitHub <https://github.com/ComputationalRadiationPhysics/picongpu/tree/dev/etc/picongpu>`_ (look for :file:`NAME.profile.example`).
The system specific directories sometimes contain a README file with some further useful information. 

For this guide we will add our scratch directory location to this profile.
Edit the profile file using your favorite editor.
If unsure use nano: ``nano ~/gpu-v100_picongpu.profile`` (save with :kbd:`Control-o`, exit with :kbd:`Control-x`).
Go to the end of the file and add a new line::

  export SCRATCH=/bigdata/hplsim/external/<username>

(Please replace ``<username>`` with your username.)

.. note::
    This is the location where runtime data and all results will be stored.
    If you're not on RoSi make sure you select the correct directory:
    Consult the documentation of your HPC cluster where to save your data.
    **On HPC clusters this is probably not your home directory.**

In the profile file you can also supply additional settings, like your email address and notification settings.

Now activate your profile::

  source ~/gpu-v100_picongpu.profile

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
  drwxr-xr-x  2 <username>    fwt   40 Nov 12 10:09 .
  drwxrwxrwt 17 root     root 400 Nov 12 10:09 ..

Create a Scenario
-----------------

As an example we will use the predefined `LaserWakefield example <https://github.com/ComputationalRadiationPhysics/picongpu/tree/dev/share/picongpu/examples/LaserWakefield>`_.
Create a directory and copy it::

  mkdir -p ~/picongpu-projects/laser-wakefield-example
  pic-create $PIC_EXAMPLES/LaserWakefield ~/picongpu-projects/laser-wakefield-example/try01
  cd ~/picongpu-projects/laser-wakefield-example/try01

Usually you would now adjust the files in the newly created directory ``~/picongpu-projects/laser-wakefield-example/try01`` – for this introduction we will use the parameters as provided.

.. note::
   The command :command:`pic-create` and the variable ``$PIC_EXAMPLES`` have been provided because you loaded the file :file:`~/gpu-v100_picongpu.profile` in the previous step.
   If this fails (printing ``pic-create: command not found``), make sure you load the PIConGPU profile by executing ``source ~/gpu-v100_picongpu.profile``.

Compile and Run
---------------

**Now use a compute node.**
Your profile provides a helper command for that::

  getDevice

(You can now run ``hostname`` to see which node you are using.)

Now build the scenario::

  # switch to the scenario directory if you haven't already
  cd ~/picongpu-projects/laser-wakefield-example/try01
  pic-build -j 6

This will take a while, go grab a coffee.
The ``-j 6`` behind ``pic-build`` tells the compiler to compile in parallel.
You can speed up the compile process by requesting more cores here. 
However this will also require more memory for the compile process.
If this fails, read the manual or ask a colleague.

After a successful build, run (still on the compute node, still inside your scenario directory)::

  tbg -s bash -t $PICSRC/etc/picongpu/bash/mpiexec.tpl -c etc/picongpu/1.cfg $SCRATCH/laser-wakefield-example/try01/run01

- :command:`tbg`: tool provided by PIConGPU
- ``bash``: the “submit system”, e.g. use ``sbatch`` for slurm
- ``$PICSRC``: the path to your PIConGPU source code, automatically set when sourcing :file:`gpu-v100_picongpu.profile`
- :file:`$PICSRC/etc/picongpu/bash/mpiexec.tpl`: options for the chosen submit system
- :file:`etc/picongpu/1.cfg`: runtime options (number of GPUs, etc.)
- :file:`$SCRATCH/laser-wakefield-example/try01/run01`: not-yet-existing destination for your result files

.. note::
   Usually you would use the *workload manager* (`SLURM <https://slurm.schedmd.com/>`_ on RoSi) to submit your jobs
   instead of running them interactively like we just did.
   You can try that with::

     # go back to the login node
     exit
     hostname
     # ...should now display one of rosi5, rosi4, ...

     # resubmit your simulation with a new directory:
     tbg -s sbatch -c etc/picongpu/1.cfg -t etc/picongpu/rosi-hzdr/gpu-v100.tpl $SCRATCH/laser-wakefield-example/try01/run02

   This will print a confirmation message (e.g. ``Submitted batch job 3769365``),
   but no output of PIConGPU itself will be printed.
   Using ``squeue -u $USER`` you can view the current status of your job.

   Note that we not only used a different "submit system" ``sbatch``,
   but also changed the template file to :file:`etc/picongpu/rosi-hzdr/gpu-v100.tpl`.
   (This template file is directly located in your project directory.)
   Both profile and template file are built for the same compute device, the NVIDIA Volta "V100" GPU.

Examine the Results
-------------------

Results are located at :file:`$SCRATCH/laser-wakefield-example/try01/run01`.

To view pretty pictures from a linux workstation you can use the following process (execute on your workstation, **not the HPC cluster**)::

  # Create a “mount point” (empty directory)
  mkdir -p ~/mnt/scratch

  # Mount the data directory using sshfs
  sshfs -o default_permissions -o idmap=user -o uid=$(id -u) -o gid=$(id -g) rosi5:DATADIR ~/mnt/scratch/

Substitute DATADIR with the full path to your data (*scratch*) directory, e.g. :file:`/bigdata/hplsim/scratch/alice`.

Browse the directory using a file browser/image viewer.
Check out :file:`~/mnt/scratch/laser-wakefield-example/try01/run01/simOutput/pngElectronsYX/` for image files.

On RoSi, you can also use the in-browser file manager to access your files.
Go to `<https://rosi.hzdr.de>`_, log in, click on the Home Directory icon under Files, and navigate to :file:`/bigdata/hplsim/scratch/<username>/laser-wakefield-example/try01/run01/simOutput/pngElectronsYX/`.

openPMD
^^^^^^^

The PNG output is great for having a quick look at the example results, but it is a rather inefficient way of generating and storing your simulation results.
In general, we write simulation data into ADIOS2 (``.bp5`` and ``.bp4`` extensions, and (deprecated) ``.bp``) or HDF5 (``.h5``) files with parallel I/O that follow the structure defined by the community openPMD standard.
The output can be configured using the :ref:`openPMD plugin <usage-plugins-openPMD>`.
The example you just run also provides some basic openPMD output that can be now found under :file:`simOutput/openPMD/`.
You should make yourself familiar with how to read and visualize such data.
Here are some useful resources:

- `openPMD standard <https://github.com/openPMD/openPMD-standard/blob/latest/STANDARD.md>`_ and the `PIC extension to the standard <https://github.com/openPMD/openPMD-standard/blob/latest/EXT_ED-PIC.md>`_
- `openPMD API docs <https://openpmd-api.readthedocs.io/en/latest/>`_ API for reading and writing openPMD output.
  Powerful tool, its proper use requires some familiarity with the standard itself.
- `openPMD-viewer <https://github.com/openPMD/openPMD-viewer>`_ Data visualization Python library built on top of the API.
- `openPMD-scipp <https://github.com/pordyna/openpmd_scipp>`_ Easy visualization of mesh data (fields) using the `plopp <https://scipp.github.io/plopp/>`_ and `scipp <https://github.com/scipp>`_ Python libraries.
- `openPMD pandas support <https://openpmd-api.readthedocs.io/en/latest/analysis/pandas.html>`_ Straightforward solution for working with particle data.
- `openpmd-ls <https://openpmd-api.readthedocs.io/en/latest/utilities/cli.html#openpmd-ls>`_ Command for getting an overview over the content of an openPMD output via command line.
- `bpls <https://adios2.readthedocs.io/en/latest/ecosystem/utilities.html#bpls-inspecting-data>`_ Powerful tool for inspecting individual ADIOS2 files via command line.
- `h5ls <https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__l_s__u_g.html>`_ and `h5dump <https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_p__u_g.html>`_: CLI tools for quick inspection of HDF5 output.
- `myHDF5 <https://myhdf5.hdfgroup.org/>`_ Easy to use Web GUI for inspection of HDF5 files based on `H5Web <https://github.com/silx-kit/h5web>`_. Also available as a `plugin for VS Code <https://github.com/silx-kit/vscode-h5web>`_.

We recommend trying out the openPMD-viewer and the openPMD-scipp tools first, followed by a quick read through the standard itself and the python part of the `first read API example <https://openpmd-api.readthedocs.io/en/latest/usage/firstread.html>`_.

When directly using the openPMD API make sure you make yourself familiar with when you need to call ``series.flush()``.
See `Flush Chunk <https://openpmd-api.readthedocs.io/en/latest/usage/firstread.html#flush-chunk>`_ and `Deferred Data API Contract <https://openpmd-api.readthedocs.io/en/latest/usage/workflow.html#deferred-data-api-contract>`_ for more details.
Forgetting it or modifying the data before flush was performed is a common user mistake that can lead to very confusing results.

In addition to the standard openPMD output PIConGPU offers some reduced diagnostics via :ref:`other plugins <usage-plugins>` including the very powerful :ref:`flexible binning plugin <usage-plugins-binningPlugin>`.
