.. _development-ci:

Continuous Integration
======================

.. sectionauthor:: René Widera

Where CI runs
-------------

The CI is a GitLab pipeline defined by ``.gitlab-ci.yml`` in the repository
root and the included files under ``share/ci/``.
There is no ``.github/`` directory in the repository.

What is tested?
---------------

The CI is compiling tests with different compilers and boost versions.
We compile examples, tests, and benchmark input sets from ``share/picongpu``.
Unit tests for PMacc from ``include/pmacc/test`` and examples under ``share/pmacc`` will be compiled and executed on the corresponding compute device.
PIConGPU unit test from ``share/picongpu/unit`` will be compiled and executed on the corresponding compute device.
The Python interface is tested by the ``pypicongpu-compiling-test`` and
``pypicongpu-end-to-end-test`` jobs, which compile PICMI setups and run an
end-to-end simulation.
The ``docs-snippets`` job executes the Python snippets, syntax-checks the bash
snippets and builds this documentation.
The ``pull-request-validation`` job runs the pre-commit checks.

Compiler chains
---------------

The CI is testing different version of

* ``nvcc`` with ``g++`` for NVIDIA CUDA compute devices
* ``clang`` with HIP for AMD ROCm compute devices
* ``clang`` as NVIDIA CUDA compiler for NVIDIA CUDA devices
* ``clang`` and ``g++`` for the serial alpaka accelerator to target x86-CPU compute devices

Test PR
-------

Pull requests will perform all tests described above.
By default PIConGPU input sets will only compile a reduced set of all setups.
Only the first input variation from ``cmakeFlags`` will be compiled
There is no guarantee that each setup is compiled with at least one version of the compiles chains above.
It is guaranteed that each compiler version of all compiler chains is compiling at least one PIConGPU input set.

Test dev branch
---------------

Merged PRs will perform more extensive tests compared to the PR tests.
Each PIConGPU input set is compiled at least with one compiler version of each compiler tool chain.

CI Control via Commit Message and PR Label
------------------------------------------

It is possible to control which compile tests will be executed for the pull request and dev branch.
The last commit of a pull request can contain a command in form of ``ci: <command>`` which will be taken into account by the CI.
A command must be the only text/string on a line, it is not case sensitive and can have spaces before or behind.

commands:

* ``ci: no-compile`` is disabling PIConGPU and PMacc compile and runtime tests
* ``ci: full-compile`` will execute for a PR all tests the CI is performing when PRs get merged to the dev branch.
* ``ci: picongpu`` only PIConGPU compile and runtime tests will be performed
* ``ci: pmacc`` only PMacc compile and runtime tests will be performed

For pull requests, the ``CI:no-compile`` label on the PR has the same effect as
the ``ci: no-compile`` command: it skips the PIConGPU and PMacc compile and
runtime tests.
The label is read by ``share/ci/pr_has_label.sh`` and checked first in
``share/ci/generate_reduced_matrix.sh``.
