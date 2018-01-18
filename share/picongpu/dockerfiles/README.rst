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

    docker pull ax3l/picongpu
    docker run --runtime=nvidia -p 2459:2459 -t ax3l/picongpu:0.3.0 lwfa
    # open firefox and isaac client

or

.. code:: bash

    singularity pull shub://ax3l/picongpu
    singularity exec --nv shub://ax3l/picongpu lwfa

Maintainer / Developer
----------------------

Build a new container image (build all dependencies).
First, set ``PICONGPU_BRANCH`` inside the ``Dockerfile`` accordingly to choose a version.
Set the same version of PIConGPU inside the ``Singularity`` bootstrap recipe.
You can also push the result to dockerhub and singularity-hub (you need an account there).

.. code:: bash

    cd ubuntu-1604

    # docker image
    docker build -t ax3l/picongpu:0.3.0 .
    # optional: push to dockerhub (needed for singularity bootstrap)
    docker login
    docker push ax3l/picongpu:0.3.0
    # optional: mark as latest release
    docker tag ax3l/picongpu:0.3.0 ax3l/picongpu:latest
    docker push ax3l/picongpu:latest

    # singularity image
    singularity create -s 4096 picongpu.img
    sudo singularity bootstrap picongpu.img Singularity
    # optional: push to singularity-hub
    singularity push picongpu.img --name ax3l/picongpu --tag 0.3.0

Recipes
-------

Currently, the following build recipes exist:

* ``ubuntu-1604/``: Ubuntu 16.04, CUDA 9.0.176 (nvcc), GCC 5.4.0
