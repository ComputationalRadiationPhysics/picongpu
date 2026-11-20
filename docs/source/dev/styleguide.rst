.. _development-styleguide:

Coding Guide Lines
==================

.. sectionauthor:: Axel Huebl

.. seealso::

   Our coding guide lines are documented in `this repository <https://github.com/ComputationalRadiationPhysics/contributing>`_.

   Installing the pre-commit hooks that run these formatters, and our
   commit-message rules, are described in
   :doc:`PIConGPU Commit Rulez <docs/COMMIT>`.

Source Style
------------

For contributions, *an ideal patch blends in the existing coding style around it* without being noticed as an addition when applied.
Nevertheless, please make sure *new files* follow the styles linked above as strict as possible from the beginning.

C++ sources are formatted with clang-format 20.1.8 (LLVM 20.1); Python sources
follow `black <https://pypi.org/project/black/>`_ with a line length of 120.
Both are automated in the pre-commit hooks.

If you are not able or willing to use ``pre-commit``, you can instead do the
following manually to get close to the same result.

For Python code, install *black* and run:

.. code-block:: bash

   black -l 120

The following describes formatting of C++ code.
Install *clang-format 20.1.8* from LLVM 20.1.

Format All Files
^^^^^^^^^^^^^^^^

To format all files in your working copy, you can run this command in bash from the root folder of PIConGPU:

.. code-block:: bash

   find include/ share/picongpu/ share/pmacc -iname "*.def" \
     -o -iname "*.h" -o -iname "*.cpp" -o -iname "*.cu" \
     -o -iname "*.hpp" -o -iname "*.tpp" -o -iname "*.kernel" \
     -o -iname "*.loader" -o -iname "*.param" -o -iname "*.unitless" \
     | xargs clang-format-20 -i

Format Only Changes, Using Git
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of using the bash command above you can use *Git* together with
*clang-format* to format your patched code only.

Before applying this command, you must extend your local git configuration **once** with all file endings used in *PIConGPU*:

.. code-block:: bash

   git config --local clangFormat.extensions def,h,cpp,cu,hpp,tpp,kernel,loader,param,unitless

Then stage your changes and run clang-format on the staged files:

.. code-block:: bash

   git clang-format

.. warning::

    On systems that install clang-format under a versioned name, use the
    corresponding git command, e.g. ``git clang-format-20``. Check that
    ``clang-format --version`` reports version 20.1.8.

The typical workflow is:

1. make your patch

2. stage the changed files in git

.. code-block:: bash

    git add <files you changed>/ -A

3. format them according to the guidelines

.. code-block:: bash

    git clang-format

4. stage the now changed (formatted) files again

.. code-block:: bash

    git add <files you changed>

5. commit the changed files

.. code-block:: bash

    git commit -m <commit message>

Please be aware that un-staged changes will not be formatted.
Formatting all changes of the previous commit can be achieved by executing the
command ``git clang-format-20 HEAD~1``.

License Header
--------------

Please **add the according license header** snippet to your *new files*:

* for PIConGPU (GPLv3+): ``src/tools/bin/addLicense <FileName>``
* for libraries (LGPLv3+ & GPLv3+):
  ``export PROJECT_NAME=PMacc && src/tools/bin/addLicense <FileName>``
* delete other headers: ``src/tools/bin/deleteHeadComment <FileName>``
* add license to all ``.hpp`` files within a directory (recursive):
  ``export PROJECT_NAME=PIConGPU && src/tools/bin/findAndDo <PATH> "*.hpp" src/tools/bin/addLicense``
* the default project name is ``PIConGPU`` (case sensitive!) and add the GPLv3+ only

Files in the directory ``thirdParty/`` are only imported from remote repositories.
If you want to improve them, submit your pull requests there and open an issue for our **maintainers** to update to a new version of the according software.
