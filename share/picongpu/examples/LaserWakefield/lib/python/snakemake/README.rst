.. _LaserWakefield-lib-python-snakemake:

Example usage of snakemake to execute PIConGPU parameter scan
--------------------------------------------------------------

`Snakemake <https://snakemake.readthedocs.io/en/stable/index.html>`_ is a python based workflow engine that can be used to automate compiling, running and post-processing of PIConGPU simulations, or any other workflow that can be represented as a directed acyclic graph (DAG). 

Each workflow consists of a ``Snakefile`` in which the workflow is defined using rules. Each rule represents a certain task. Dependencies between rules are defined by input and output files. Each rule can consist out of a shell command, python command or external python scripts (apparently also Rust, R, Julia and JupyterNB are supported). 


""""""""""
How to use
""""""""""

The example presented here performs several simulations with given input parameters.

#. Make shure you have ``snakemake`` and the ``snakemake-executor-plugin-slurm`` installed and activated.

#. Adjust the profile ``config.yaml``:
    * Define your input parameters in a csv file.

    .. warning::
        
        Snakemake will automatically perform a parameter dependend compile using CMAKE flags **if and only if** the parameter names in the header of the csv file match those in the ``.param`` file of the PIConGPU project.

    * Specify the path of your PIConGPU project, i.e. the directory where ``pic-create`` will be executed.
    * Specify the path to your PIConGPU profile and the name of your cfg file.
    * Optional: Adjust resources and other workflow parameters (see Fine-Tuning).

#. Start the workflow in the directory where the ``Snakefile`` and ``config.yaml`` are located via  

.. code-block:: shell

    snakemake --profile .

.. note::  
    You may want to start your snakemake workflow in a `screen <https://wiki.ubuntuusers.de/Screen/>`_-session.

~~~~~~~~~~~
Fine-Tuning
~~~~~~~~~~~

There are several command line options you can use to customise the behaviour of your workflow. An overview can be found in the `documentation <https://snakemake.readthedocs.io/en/stable/executing/cli.html>`_ or by using ``snakemake --help``. Here are some recomendations:

* ``--jobs N``, ``-j N``
    * Use a maximum of N jobs in parallel. Set to ``unlimited`` to allow any number of jobs.
* ``--groups``:
    * By default, each rule/task is run in a single (cluster) job. To run multiple tasks in one job, define groups in the ``Snakefile`` or ``config.yaml``, which only works if the grouped tasks are connected in the DAG.
    * In this example, the ``compile`` rule is placed in the "compile" group, so it is possible to run multiple compile processes in a single Slurm job. 
* ``--group-components``
    * Indicates how many tasks in a group will be executed in a cluster job.
    * In this example, by ``group-components: "compile=2"`` defines that 2 compile processes will be run in one slurm job.
    * This is particularly useful for smaller rules such as python post-processing, where it would be easy to have hundreds of small fast cluster jobs if no grouping took place.
* ``--dry-run``, ``-n`` 
    * Does not execute anything.
    * Useful for checking that the workflow is set up correctly and that only the desired rules are executed.
    * This is important to ensure that data that has already been written is not erased, because snakemake will re-run jobs if code or input has changed, and will erase the output of the rule before doing so. (In short, if you decide to change a path or some code in the Snakefile, you might re-run expensive simulations).
    * To prevent simulations from being repeated for the wrong reasons, use:
* ``--rerun-triggers {code,input,mtime,params,software-env}``
    * Define what triggers the rerunning of a job. By default, all triggers are used, which guarantees that results are consistent with the workflow code and configuration.
* ``--retries N``
    * Retries a failed rule N times. 
    * Can be defined for each rule individually.
    * Also useful if a cluster has a limited walltime and the picongpu flag ``--try.restart`` is to be used. Since snakemake resubmits the "submit.start", the simulation will start from the last available checkpoint, when this flag is used. 
* ``--latency-wait SECONDS```
    * Wait given SECONDS if an output file of a job is not present after the job finished. This helps if your filesystem suffers from latency (default 5).



~~~~~~~~~~~~~~~~~~~~~~~~
Resulting file structure
~~~~~~~~~~~~~~~~~~~~~~~~

The output produced by the workflow is stored in three directories next to the ``Snakefile``.

* "simualtions"
    * Contains simulation directories.
    * The name of the simulation directory is ``sim_{paramspace.wildcard_pattern}``, where ``paramspace.wildcard_pattern`` becomes, for example, ``LASERA0-4.0_PULSEDURATION-1.5e-14``.

* "simulated"
    * Contains txt files indicating whether a simulation has already run and the job id of the simulations on the cluster.

* "projects"
    * Contains the input directories of the simulations.

If you want to change the file structure, you need to change that in the ``Snakefile``.
Be aware that paths defined in your ``Snakefile`` are always relative to the location of the ``Snakefile``.

""""""""""""
What it does
""""""""""""

The workflow takes input parameters, performs a parameter dependent compile and submits the simulation to the cluster. These steps are defined as so called rules in the ``Snakefile``. The order in which the rules are executed is defined by the input and the output of the rules. This means that a rule is only executed if it's output is needed as input by another rule.

Details of the individual rules:

|

* rule all:
    * Is the so-called target rule. By default, Snakemake will only execute the very first rule specified in the ``Snakefile``. Therefore this pseudo-rule should contain all the anticipated output as its input. Snakemake will then try to generate this input.


* rule build_command:
    * Is a helper ruler that generates a string that is later used by the ``pic-build`` command and contains the information about the CMAKE flags.


* rule compile:
    * Clones the in the ``config.yaml`` defined PIConGPU project using ``pic-create``.
    * Since Snakemake relies on files to check dependencies between tasks, and a simulation has no predefined unique output file, the tpl file is modified such that it creates a unique output file, called ``finished_{params.name}.txt`` when the simulation is finished.
    * Compiles for each parameter set and then creates a simulation directory.


* rule simulate:
    * To use the ``tbg`` interface the rule simulate is a local rule.
    * The output file ("simulated/finished_{paramspace.wildcard_pattern}.txt") is created after the simulation but the shell script would be immediately done after submitting the simulation. If the task is done and the output file is not created an error occurs and the workflow fails. In order to make Snakemake wait till the simulation is finished, the status of the slurm job is checked every two minutes.
    * This control loop is set up in such a way that even if the snakemake session is aborted or fails, it will catch up with simulations already running when snakemake is restarted.

|

Using the example ``Snakefile`` and ``params.csv``, the resulting DAG looks like this.

.. image:: dag.png

""""""""""""""""""""""
Python post-processing
""""""""""""""""""""""

automatically post-process all your simulations, you can do this by adding new rules to the ``Snakefile``. 
Here is an example of what this might look like for a Python script called ``post_processing.py``:

.. code-block:: python

    rule post_processing:
        input:
            rules.simulate.output
        output:
            f"results/post_processing_{paramspace.wildcard_pattern}.png"
        params:
            sim_dir=f"simulations/sim_{paramspace.wildcard_pattern}/simOutput/openPMD", # simulation directory
            sim_params=paramspace.instance, # dictionary of parameters to generate this simulation
            generic_parameter = 1000
        script:
            "post_processing.py"

The parameters set with the ``params`` keyword, can be accessed in your python script via `snakemake.params[i]` or `snakemake.params.name_of_param`. 
Accordingly one can use ``snakemake.input`` or ``snakemake.output``.

To perform this evaluation on the cluster, add the required resource to the "config.yaml". For example, like this:

.. code-block:: yaml

    set-resources:
      post_processing: # resources for post processing
        slurm_partition: "defq"
        runtime: 20
        nodes: 1
        ntasks: 1
        mem_mb: 5000 


Then you only need to alter the input of the target rule ``all``:

.. code-block:: python

    rule all:
        input: expand("results/post_processing_{params}.png", params=paramspace.instance_patterns)

Of course you can have as many rules as you want after the simulation, just make sure that Snakemake can build a rule graph by going from the output of one rule to the input of the next rule, ending at the input of the target rule ``all``.

.. note::

    Note the ``expand()`` function in the ``all`` rule. This can be used to declare that all instances of the parameter space are meant. Further information can be found `here <https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#the-expand-function>`_.

""""""""""""""""""""""""""""
Running on a generic cluster
""""""""""""""""""""""""""""

If you want to run on a cluster other than hemera that doesn't use the slurm scheduler, check the `snakemake plugin catalog <https://snakemake.github.io/snakemake-plugin-catalog/index.html>`_ if there is an executor plugin for your batch system.
If there is no executor plugin for your batch system, you can use the generic cluster execution. 

.. warning::

    In any case, the ``Snakefile`` must be adapted to the specific cluster. 


The "Snakefile_LSF" is an example for running on a LSF cluster (e.g. Summit) using the generic cluster executer. 

To use it:
    * Install  ``snakemake-executor-plugin-cluster-generic`` plugin.
    * Adapt the executor and add submit command in the ``config.yaml``:

.. code-block:: yaml

    executor: cluster-generic
    cluster-generic-submit-cmd: "'bsub -P {resources.proj} -nnodes {resources.nodes} -W {resources.walltime}'"
    set-resources:
      compile: # define resources for picongpu compile
        proj: "csc999" # change to your project!
        walltime: 120
        nodes: 1

* Start workflow with

.. code-block:: shell

    snakemake --profile .

.. note::

    Recently an `LSF executor plugin <https://github.com/BEFH/snakemake-executor-plugin-lsf>`_ has been developed which has not been tested with the PIConGPU workflow. If you have access to a LSF cluster, give it a try. 
