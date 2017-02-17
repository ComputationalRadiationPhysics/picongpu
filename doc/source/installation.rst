Installation
============

Depending on your experience, role, computing environment and expectations optimal hardware utilization, you have several ways to install the dependencies required to run PIConGPU.
Chose your favorite install path below, young padavan:

HPC Cluster
-----------

SysAdmin
~~~~~~~~

- use `spack`
- build from source and provide `lmod` modules

User
~~~~

- use `spack`
- build from source

Desktop
-------

Root/Admin
~~~~~~~~~~

- `apt-get install` what is possible
- use `spack`
- build from source

User
~~~~

- use `spack`
- use `nvidia-docker`
- build from source

Cloud
-----

AWS
~~~

- use `spack`
- use `nvidia-docker`
- build from source

Google Cloud
~~~~~~~~~~~~

- use `spack`
- use `nvidia-docker`
- build from source



bli ...

.. code::

    conda install -c conda-forge ...

bla ...

.. code::

    cmake -D CMAKE_INSTALL_PREFIX=your_install_prefix
    make install

