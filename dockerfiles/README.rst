Dockerfiles
===========

User
----

Download a docker container and execute it locally.
Opens a bash prompt with a fully set up PIConGPU environment.

.. code:: bash

    nvidia-docker pull ax3l/picongpu
    nvidia-docker run -it ax3l/picongpu

Alternatively, run an already build example.
This exposes the ISAAC port to connect via the webclient to.

.. code:: bash

    nivida-docker pull ax3l/picongpu
    nvidia-docker run -d ax3l/picongpu -p 2459:2459 isaac
    # in a second terminal
    nvidia-docker run -it ax3l/picongpu start_lwfa.sh
    # open a browser for the ISAAC client ...

Maintainer / Developer
----------------------

Build a new container image (build all dependencies).
First, set ``PICONGPU_BRANCH`` inside the ``Dockerfile`` accordingly to choose a version.
You can also push the result to dockerhub (you need an account there).

.. code:: bash

    cd ubuntu-1604

    # docker image
    docker build -t ax3l/picongpu:0.3.0 .
    # optional: push to dockerhub
    docker login
    docker push ax3l/picongpu:0.3.0
    # optional: mark as latest release
    docker tag ax3l/picongpu:0.3.0 ax3l/picongpu:latest
    docker push ax3l/picongpu:latest

Recipes
-------

Currently, the following build recipes exist:

* ``ubuntu-1604/``: Ubuntu 16.04, CUDA 8.0.61 (nvcc), GCC 5.4.0
