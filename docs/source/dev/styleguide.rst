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

Unfortunately, we currently do not have tools available to auto-format all aspects of our style guidelines.
Since we want to focus on the content of your contribution, we try to cover as much as possible by automated tests which you always have to pass.
Nevertheless, we will not enforce the still uncovered, *non-semantic aspects* of style in a *pedantic* way until we find a way to automate it fully.

(That also means that we do not encourage manual style-only changes of our existing code base, since both you and us have better things to do than adding newlines and spaces manually.
Doxygen and documentation additions are always welcome!)

License Header
--------------

Please **add the according license header** snippet to your *new files*:

* for PIConGPU (GPLv3+): ``src/tools/bin/addLicense <FileName>``
* for libraries (LGPLv3+ & GPLv3+):
  ``export PROJECT_NAME=libPMacc && src/tools/bin/addLicense <FileName>``
* delete other headers: ``src/tools/bin/deleteHeadComment <FileName>``
* add license to all ``.hpp`` files within a directory (recursive):
  ``export PROJECT_NAME=PIConGPU && src/tools/bin/findAndDo <PATH> "*.hpp" src/tools/bin/addLicense``
* the default project name is ``PIConGPU`` (case sensitive!) and add the GPLv3+ only

Files in the directory ``thirdParty/`` are only imported from remote repositories.
If you want to improve them, submit your pull requests there and open an issue for our **maintainers** to update to a new version of the according software.
