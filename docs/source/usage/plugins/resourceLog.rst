.. _usage-plugins-resourceLog:

Resource Log
------------

Writes resource information such as rank, position, current simulation step, particle count, and cell count
as json or xml formatted string to output streams (file, stdout, stderr).

.cfg file
^^^^^^^^^

Run the plugin for each nth time step: ``--resourceLog.period n``

The following table will describes the settings for the plugin:

============================ ===================================================================================
Command line option          Description
============================ ===================================================================================
``--resourceLog.properties`` Selects properties to write [rank, position, currentStep, particleCount, cellCount]
``--resourceLog.format``     Selects output format [json, jsonpp, xml, xmlpp]
``--resourceLog.stream``     Selects output stream [file, stdout, stderr]
``--resourceLog.prefix``     Selects the prefix for the file stream name
============================ ===================================================================================

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

no extra allocation.

Host
""""

negligible.

Output / Example
^^^^^^^^^^^^^^^^

Using the options

.. code:: bash

   --resourceLog.period 1 \
   --resourceLog.stream stdout \
   --resourceLog.properties rank position currentStep particleCount cellCount \
   --resourceLog.format jsonpp

will write resource objects to stdout such as:

.. code::

    [1,1]<stdout>:    "resourceLog": {
    [1,1]<stdout>:        "rank": "1",
    [1,1]<stdout>:        "position": {
    [1,1]<stdout>:            "x": "0",
    [1,1]<stdout>:            "y": "1",
    [1,1]<stdout>:            "z": "0"
    [1,1]<stdout>:        },
    [1,1]<stdout>:        "currentStep": "357",
    [1,1]<stdout>:        "cellCount": "1048576",
    [1,1]<stdout>:        "particleCount": "2180978"
    [1,1]<stdout>:    }
    [1,1]<stdout>:}

For each format there exists always a non pretty print version to simplify further processing:

.. code::

    [1,3]<stdout>:{"resourceLog":{"rank":"3","position":{"x":"1","y":"1","z":"0"},"currentStep":"415","cellCount":"1048576","particleCount":"2322324"}}
