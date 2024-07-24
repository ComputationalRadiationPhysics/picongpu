.. _PlantUML_ŕeference:

PlantUML
========

`PlantUML <https://plantuml.com/>`_ is a java tool generating UML, and similar, diagrams from text files and has been used to generate UML diagrams in our documentation.
For each UML diagram generated using PlantUML we also provide the source file ``<figure_name>.txt``.

Install PlantUML
----------------
To install PlantUMl you need to

- install a compatible java version
- download ``plantUML.jar`` from the `PlantUML website <https://plantuml.com/download>`_ and place it in a folder of your choice.

Updating Figures
----------------

To update a figure edit the ``<figure_name>.txt`` file and regenerate the figure by running the following command in the folder containing the ``plantUML.jar`` file.

.. code:: bash

  java -DPLANTUML_LIMIT_SIZE=8192 -jar plantuml.jar

This will open a GUI allowing you to choose a directory in which PlantUML will try to regenerate all figures from the contained source files.

.. note::

  UML source files are identified by PlantUML by the @startuml line in somewhere the file.
