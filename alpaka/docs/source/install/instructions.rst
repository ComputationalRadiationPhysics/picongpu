.. highlight:: bash

alpaka Installation
===================

* Clone alpaka from github.com

.. code-block::

  git clone https://github.com/alpaka-group/alpaka
  cd alpaka

* Install alpaka

.. code-block::

  # git clone https://github.com/alpaka-group/alpaka
  # cd alpaka
  mkdir build && cd build
  cmake -DCMAKE_INSTALL_PREFIX=/install/ ..
  cmake --install .

* Configure Accelerators

.. code-block::

  # ..
  cmake -DALPAKA_ACC_GPU_CUDA_ENABLE=ON ..

* Build an example

.. code-block::

  # ..
  cmake -Dalpaka_BUILD_EXAMPLES=ON ..
  make vectorAdd
  ./example/vectorAdd/vectorAdd # execution

* Build and run tests

.. code-block::

  # ..
  cmake -DBUILD_TESTING=ON ..
  make
  ctest
