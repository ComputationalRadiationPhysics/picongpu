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

Format Changes Using Git
^^^^^^^^^^^^^^^^^^^^^^^^

Instead of using the bash command above you can use *Git* together with *ClangFormat* to format your patched code only.
Before applying this command, you must extend your local git configuration **once** with all file endings used in *PIConGPU*:

.. code-block:: bash

   git config --local clangFormat.extensions def,h,cpp,cu,hpp,tpp,kernel,loader,param,unitless

.. warning::

    The binary for *ClangFormat* is on some operating systems called `clang-format`.
    If so please check that `clang-format --version` returns version `11.X.X`.

For only formatting lines you added using `git add`, call `git clang-format-11` before you create a commit.
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
