.. _PlantUML_ŕeference:

PlantUML
========

`PlantUML <https://plantuml.com/>`_ is a java tool generating UML, and similar, diagrams from text files and has been used to generate UML diagrams in our documentation.
For each UML diagram generated using PlantUML we also provide the source file ``<figure_name>.txt``.

Install PlantUML
----------------

See `PlantUML documentation <https://plantuml.com/starting>`_ for detailed instructions.

TLDR:
^^^^^

- install a compatible ``java`` version, for example: :code:`sudo apt update && sudo apt install openjdk-21-jre-headless`
- download ``plantUML.jar`` from the `PlantUML website <https://plantuml.com/download>`_ and place it in a folder of your choice.

Updating Figures
----------------

To update a figure, edit the ``<figure_name>.txt`` file, and regenerate the figure.

To generate a PNG from an existing PlantUML source file run the following command in the folder containing the ``plantUML.jar`` file,

.. code:: bash

  java -DPLANTUML_LIMIT_SIZE=8192 -jar plantuml.jar

and select the directory containing the PlantUML source file in the GUI.
This will generate a PNG file from every PlantUML source file in the selected directory

.. note::

  UML source files are identified by PlantUML by the string ``@startuml`` in somewhere the file.
