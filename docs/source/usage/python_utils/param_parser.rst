.. _usage-python-utils-param-parser:

Parameter File Parser
---------------------

.. sectionauthor:: Julian J. Lenz

The ``param_parser`` module reads the JSON parameter files produced by the PIConGPU parameter-scan tooling and turns them into command-line arguments.
Such a file maps parameter names to a ``type`` (``compile`` or ``run``) and a list of ``values``.
``parse`` extracts the first value of every parameter of the requested type and emits either the ``-DPARAM_OVERWRITES:LIST='...'`` argument understood by ``pic-configure -c`` (for ``compile`` parameters) or plain ``<name>=<value>`` pairs for ``tbg -o`` (for ``run`` parameters).

.. code:: python

   from picongpu.extra.utils import param_parser

   values = param_parser.read_range_file("params.json")          # {name: [values]}
   param_parser.parse("params.json", "compile")                  # -DPARAM_OVERWRITES:LIST='...'
   param_parser.parse("params.json", "run")                      # -o 'name=value ...'

The module also provides a small script interface, ``python param_parser.py -t <compile|run> -i <file>``, that prints the parsed argument string.

.. automodule:: picongpu.extra.utils.param_parser
   :members: read_range_file, to_macro_name, parse

See :ref:`usage-parameter_scans` for the scan workflows that consume these arguments.
