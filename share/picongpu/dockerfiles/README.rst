Dockerfiles
===========

User
----

Download a docker container and execute it locally.
Opens a bash prompt with a fully set up PIConGPU environment.

.. code:: bash

    nvidia-docker pull ax3l/picongpu
    nvidia-docker run -it ax3l/picongpu

or in singularity:

.. code:: bash

    # singularity pull ax3l/picongpu
    singularity run --nv picongpu.img

Alternatively, run an already configured and fully build example.
This exposes the ISAAC port to connect via the webclient to.

.. code:: bash

    nivida-docker pull ax3l/picongpu
    nvidia-docker run -p 2459:2459 -t ax3l/picongpu:0.3.0 /bin/bash -lc start_lwfa
    # open firefox and isaac client

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

Recipes
-------

Currently, the following build recipes exist:

* ``ubuntu-1604/``: Ubuntu 16.04, CUDA 8.0.61 (nvcc), GCC 5.4.0
