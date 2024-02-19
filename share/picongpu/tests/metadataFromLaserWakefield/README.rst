.. _tests-metadataFromLaserWakefield:

Metadata From the Laser Wakefield Example
=================================================================

.. sectionauthor:: Julian Lenz
.. moduleauthor:: Julian Lenz

This example compiles the `LaserWakefield` example from the `share/picongpu/examples` folder and runs it with

.. literalinclude:: bin/ci.sh
   :language: bash
   :start-after: doc-include-start: cmdline
   :end-before: doc-include-end: cmdline
   :dedent:

such that it outputs its metadata to `picongpu-metadata.json`. This file is compared with the reference file
`picongpu-metadata.json.reference`.
