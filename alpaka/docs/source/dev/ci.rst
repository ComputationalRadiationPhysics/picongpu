Continuous Integration
======================

For automatic testing we use two different systems: ``GitHub Actions`` and ``GitLab CI``. ``GitHub Actions`` are used for a wide range of build tests and also some CPU runtime tests. ``GitLab CI`` allows us to run runtime tests on GPUs and CPU architectures other than x86, like ARM or IBM POWER.

GitHub Actions
----------------

The configuration of ``GitHub Actions`` can be found in the ``.github/workflows/`` folder. This CI uses unmodified containers from ``Docker Hub`` and sets up the environment during the test job. A caching mechanism speeds up the job times. The scripts for setting up the environment, building alpaka and running test are located in the ``script/`` folder.

GitLab CI
---------

We use ``GitLab CI`` because it allows us to use self-hosted system, e.g. GPU systems. The GitHub repository is mirrored on https://gitlab.com/hzdr/crp/alpaka . Every commit or pull request is automatically mirrored to GitLab and triggers the CI. The configuration of the ``GitLab CI`` is stored in the file ``.gitlab-ci.yml``. The workflow of a ``GitLab CI`` is different from ``GitHub Actions``. Instead of downloading an unmodified container from ``Docker Hub`` and preparing the environment during the test job, ``GitLab CI`` uses containers which are already prepared for the tests. The containers are built in an `extra repository <https://gitlab.hzdr.de/crp/alpaka-group-container>`_ and contain all dependencies for alpaka. All available containers can be found `here <https://gitlab.hzdr.de/crp/alpaka-group-container/container_registry>`_. The scripts to build alpaka and run the tests are shared with ``GitHub Actions`` and located at ``script/``.

.. figure:: /images/arch_gitlab_mirror.svg
   :alt: Relationship between GitHub.com, GitLab.com and HZDR gitlab-ci runners

   Relationship between GitHub.com, GitLab.com and HZDR gitlab-ci runners

To change how the tests are built and executed, modify the code in the `alpaka repository <https://github.com/alpaka-group/alpaka>`_. If the container environment with the dependencies needs to be changed, please open an issue or contribute to `alpaka-group-container repository <https://gitlab.hzdr.de/crp/alpaka-group-container>`_.
