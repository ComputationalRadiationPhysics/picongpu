Automatic Testing
=================

For automatic testing we use two different systems: ``GitHub Actions`` and ``GitLab CI``. ``GitHub Actions`` are used for a wide range of build tests and also some CPU runtime tests. ``GitLab CI`` allows us to run runtime tests on GPUs and CPU architectures other than x86, like ARM or IBM POWER.

GitHub Actions
----------------

The configuration of ``GitHub Actions`` can be found in the ``.github/workflows/`` folder. This CI uses unmodified containers from ``Docker Hub`` and sets up the environment during the test job. A caching mechanism speeds up the job times. The scripts for setting up the environment, building alpaka and running test are located in the ``script/`` folder.

clang-format
++++++++++++

The first CI job run is clang-format, which will verify the formatting of your changeset.
Only of this check passes, will the remainder of the GitHub CI continue.
In case of a formatting failure, a patch file is attached as an artifact to the GitHub action run.
You can apply this patch file to your changeset to fix the formatting.

GitLab CI
---------

We use ``GitLab CI`` because it allows us to use self-hosted system, e.g. GPU systems.
The GitHub repository is mirrored on https://gitlab.com/hzdr/crp/alpaka .
Every commit or pull request is automatically mirrored to GitLab and triggers the CI.
The configuration of the ``GitLab CI`` is stored in the file ``.gitlab-ci.yml``.
The workflow of a ``GitLab CI`` is different from ``GitHub Actions``.
Instead of downloading an unmodified container from ``Docker Hub`` and preparing the environment during the test job, ``GitLab CI`` uses containers which are already prepared for the tests.
The containers are built in an `extra repository <https://gitlab.hzdr.de/crp/alpaka-group-container>`_ and contain all dependencies for alpaka.
All available containers can be found `here <https://gitlab.hzdr.de/crp/alpaka-group-container/container_registry>`_.
The scripts to build alpaka and run the tests are shared with ``GitHub Actions`` and located at ``script/``.

Most of the jobs for the GitLab CI are generated automatically.
For more information, see the section :ref:`The Job Generator`.

It is also possible to define custom jobs, see :ref:`Custom jobs`.

.. only:: html

  .. figure:: /images/arch_gitlab_mirror.svg
     :alt: Relationship between GitHub.com, GitLab.com and HZDR gitlab-ci runners

     Relationship between GitHub.com, GitLab.com and HZDR gitlab-ci runners

.. only:: latex

  .. figure:: /images/arch_gitlab_mirror.png
     :alt: Relationship between GitHub.com, GitLab.com and HZDR gitlab-ci runners
  
     Relationship between GitHub.com, GitLab.com and HZDR gitlab-ci runners

The Container Registry
++++++++++++++++++++++

Alpaka uses containers in which as many dependencies as possible are already installed to save job execution time.
The available containers can be found `here <https://gitlab.hzdr.de/crp/alpaka-group-container/container_registry>`_.
Each container provides a tool called ``agc-manager`` to check if a software is installed. The documentation for ``agc-manager`` can be found `here <https://gitlab.hzdr.de/crp/alpaka-group-container/-/tree/master/tools>`_.
A common way to check if a software is already installed is to use an ``if else statement``.
If a software is not installed yet, you can install it every time at job runtime.

.. code-block:: bash

 if agc-manager -e boost@${ALPAKA_CI_BOOST_VER} ; then
   export ALPAKA_CI_BOOST_ROOT=$(agc-manager -b boost@${ALPAKA_CI_BOOST_VER})
 else
   # install boost
 fi

This statement installs a specific boost version until the boost version is pre-installed in the container.
To install a specific software permanently in the container, please open an issue in the `alpaka-group-container repository <https://gitlab.hzdr.de/crp/alpaka-group-container/-/issues>`_.

The Job Generator
+++++++++++++++++

Alpaka supports a large number of different compilers with different versions and build configurations.
To manage this large set of possible test cases, we use a job generator that generates the CI jobs for the different compiler and build configuration combinations.
The jobs do not cover all possible combinations, as it would be too much to run the entire CI pipeline in a reasonable amount of time.
Instead, the job generator uses `pairwise testing <https://en.wikipedia.org/wiki/All-pairs_testing>`_.

The stages of the job generator are:

.. only:: html

  .. figure:: /images/job_generator_flow.svg
     :alt: workflow fo the CI job generator

.. only:: latex

  .. figure:: /images/job_generator_flow.png
     :alt: workflow fo the CI job generator

The job generator is located at `script/job_generator/ <https://github.com/alpaka-group/alpaka/tree/develop/script/job_generator/>`_.
The code is split into two parts. One part is alpaka-specific and stored in this repository.
The other part is valid for all alpaka-based projects and stored in the `alpaka-job-coverage library <https://pypi.org/project/alpaka-job-coverage/>`_.

Run Job Generator Offline
*************************

First you need to install the dependencies.
It is highly recommended to use a virtual environment.
You can create one for example with the `venv <https://docs.python.org/3/library/venv.html>`_-Python module or with `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.
Once you have created a virtual environment, you should activate it and install the Python packages via:

.. code-block:: bash

 pip install -r script/job_generator/requirements.txt

After installing the Python package, you can simply run the job generator via:

.. code-block:: bash

 # 3.0 is the version of the docker container image
 # run `python ci/job_generator/job_generator.py --help` to see more options
 python script/job_generator/job_generator.py 3.0

The generator creates a ``jobs.yaml`` in the current directory with all job combinations.

Filter and Reorder Jobs
***********************

The job generator provides the ability to filter and reorder the generated job matrix using `Python <https://docs.python.org/3/howto/regex.html>`_ regex.
The regex is applied via the commit message for the current commit:

.. code-block::

  Add function to filter and reorder CI jobs

  This commit message demonstrates how it works. The job filter removes
  all jobs whose names do not begin with NVCC or GCC. Then the jobs are
  reordered. First all GCC11 are executed, then all GCC8 and then the
  rest.

  CI_FILTER: ^NVCC|^GCC
  CI_REORDER: ^GCC11 ^GCC8

The job generator looks for a line starting with the prefix ``CI_FILTER`` to filter the jobs or ``CI_REORDER`` to reorder the jobs.
The filter statement is a single regex.
The reorder statement can consist of multiple regex separated by a whitespace.
For reordering, the jobs have the same order as the regex.
This means that all orders matching the first regex are executed first, then the orders matching the second regex and so on.
At the end, all orders that do not match any regex are executed.
**Attention:** the order is only guaranteed across waves.
Within a wave, it is not guaranteed which job will start first.

It is not necessary that both prefixes are used.
One of them or none is also possible.

.. hint::

  You can test your regex offline before creating and pushing a commit. The ``job_generator.py`` provides the ``--filter`` and ``--reorder`` flags that do the same thing as the lines starting with ``CI_FILTER`` and ``CI_REORDER`` in the commit message.

.. hint::

  Each time the job generator runs it checks whether the container images exist. This is done by a request to the container registry which takes a lot of time. Therefore you can skip the check with the ``--no-image-check`` argument to speed up checking filters and reordering regex strings.

Develop new Feature for the alpaka-job-coverage Library
*******************************************************

Sometimes one needs to implement a new function or fix a bug in the alpaka-job-coverage library while they are implementing a new function or fixing a bug in the alpaka job generator.
Affected filter rules can be recognized by the fact that they only use parameters defined in this `globals.py <https://github.com/alpaka-group/alpaka-job-matrix-library/blob/main/src/alpaka_job_coverage/globals.py>`_.

The following steps explain how to set up a development environment for the alpaka-job-coverage library and test your changes with the alpaka job generator.

We strongly recommend using a Python virtual environment.

.. code-block:: bash

 # if not already done, clone repositories
 git clone https://github.com/alpaka-group/alpaka-job-matrix-library.git
 git clone https://github.com/alpaka-group/alpaka.git

 cd alpaka-job-matrix-library
 # link the files from the alpaka-job-matrix-library project folder into the site-packages folder of your environment
 # make the package available in the Python interpreter via `import alpaka_job_coverage`
 # if you change a src file in the folder, the changes are immediately available (if you use a Python interpreter instance, you have to restart it)
 python setup.py develop
 cd ..
 cd alpaka
 pip install -r script/job_generator/requirements.txt

Now you can simply run the alpaka job generator.
If you change the source code in the project folder alpaka-job-matrix-library, it will be immediately available for the next generator run.

Custom jobs
+++++++++++

You can create custom jobs that are defined as a yaml file.
You can add the path of the folder to the function ``add_custom_jobs()`` in ``script/job_generator/custom_job.py``.
The function automatically read all files in the folder, which matches a filter function and loads the GitLab CI jobs.
The custom jobs are added to the same job list as the generated jobs and distributed to the waves.
