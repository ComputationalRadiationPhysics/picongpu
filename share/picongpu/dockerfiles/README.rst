Dockerfiles
===========

User
----

Download a docker container and execute it locally.
Opens a bash prompt with a fully set up PIConGPU environment.

.. code:: bash

    docker pull ax3l/picongpu
    docker run --runtime=nvidia -it ax3l/picongpu

or in singularity:

.. code:: bash

    singularity pull shub://ax3l/picongpu
    singularity shell --nv shub://ax3l/picongpu

Alternatively, run an already configured and fully build example.
This exposes the ISAAC port to connect via the webclient to.

.. code:: bash

    docker pull computationalradiationphysics/picongpu
    docker run --runtime=nvidia -p 2459:2459 -t computationalradiationphysics/picongpu:0.6.0-rc1
    # open firefox and isaac client

or

.. code:: bash

    singularity pull shub://ax3l/picongpu
    singularity exec --nv shub://ax3l/picongpu lwfa_live

.. note::

   PIConGPU is perfectly multi-GPU capable and scales up to thousands of GPUs on the largest GPU clusters available.
   In order to share data between ranks, the communication layer we use (MPI) requires shared system memory for IPC and pinned (page-locked) system memory.
   The default docker limits on these resources are very small (few dozen MB) and need to be increased in order to run on multiple GPUs.

   For the ``docker run`` commands above, append: ``--shm-size=1g --ulimit memlock=-1`` to increase the defaults.

Maintainer / Developer
----------------------

Build a new container image (build all dependencies).
First, set ``PICONGPU_BRANCH`` inside the ``Dockerfile`` accordingly to choose a version.
Set the same version of PIConGPU inside the ``Singularity`` bootstrap recipe.
You can also push the result to dockerhub and singularity-hub (you need an account there).

.. code:: bash

    cd ubuntu-2004

    # docker image
    docker build -t computationalradiationphysics/picongpu:0.6.0-rc1
    # optional: push to dockerhub (needed for singularity bootstrap)
    docker login
    docker push computationalradiationphysics/picongpu:0.6.0-rc1
    # optional: mark as latest release
    docker tag computationalradiationphysics/picongpu:0.6.0-rc1
    docker push computationalradiationphysics/picongpu:latest

    # singularity image
    singularity create -s 4096 picongpu.img
    sudo singularity bootstrap picongpu.img Singularity
    # optional: push to a singularity registry
    # setup your $HOME/.sregistry first
    sregistry push picongpu.img --name ax3l/picongpu --tag 0.6.0-rc1

Recipes
-------

Currently, the following build recipes exist:

* ``ubuntu-2004/``: Ubuntu 20.04, CUDA 11.2 (nvcc), GCC 9.3.0
