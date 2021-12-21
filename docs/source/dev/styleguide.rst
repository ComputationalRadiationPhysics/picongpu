.. _development-styleguide:

Coding Guide Lines
==================

.. sectionauthor:: Axel Huebl

.. seealso::

   Our coding guide lines are documented in `this repository <https://github.com/ComputationalRadiationPhysics/contributing>`_.

Source Style
------------

For contributions, *an ideal patch blends in the existing coding style around it* without being noticed as an addition when applied.
Nevertheless, please make sure *new files* follow the styles linked above as strict as possible from the beginning.

clang-format-11 should be used to format the code.
There are different ways to format the code.

Format All Files
^^^^^^^^^^^^^^^^

To format all files in your working copy, you can run this command in bash from the root folder of PIConGPU:

.. code-block:: bash

   find include/ share/picongpu/ share/pmacc -iname "*.def" \
     -o -iname "*.h" -o -iname "*.cpp" -o -iname "*.cu" \
     -o -iname "*.hpp" -o -iname "*.tpp" -o -iname "*.kernel" \
     -o -iname "*.loader" -o -iname "*.param" -o -iname "*.unitless" \
     | xargs clang-format-11 -i

Format Only Changes, Using Git
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of using the bash command above you can use *Git* together with *ClangFormat* to format your patched code only.

    *ClangFormat* is an external tool for code formating that can be called by *Git* on changed files only and
    is part of clang tools.

Before applying this command, you must extend your local git configuration **once** with all file endings used in *PIConGPU*:

.. code-block:: bash

   git config --local clangFormat.extensions def,h,cpp,cu,hpp,tpp,kernel,loader,param,unitless

After installing, or on a cluster loading the module(see introduction), clangFormat can be called by git on all **staged files** using the command:

.. code-block:: bash

   git clangFormat

.. warning::

    The binary for *ClangFormat* is called `clang-format` on some operating systems.
    If *clangFormat* is not recognized, try *clang-format* instead, in addition please check that `clang-format --version` returns version `11.X.X` in this case.

The Typical workflow using git clangFormat is the following,

1. make your patch

2. stage the changed files in git

.. code-block:: bash

    git add <files you changed>/ -A

3. format them according to guidelines

.. code-block:: bash

    git clangFormat

4. stage the now changed(formated) files again

.. code-block:: bash

    git add <files you changed>

5. commit changed files

.. code-block:: bash

    git commit -m <commit message>

Please be aware that un-staged changes will not be formatted.
Formatting all changes of the previous commit can be achieved by executing the command `git clang-format-11 HEAD~1`.

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
