.. _serialization:

(De-)Serialization and Reproducibility
======================================

Every element of the interface and middle layer is a
`Pydantic <https://docs.pydantic.dev/>`__ model.
This provides automatic validation and (de-)serialization,
which you can use to save, inspect and reuse simulations.

Serializing a simulation
------------------------

``simulation.get_as_pypicongpu()`` returns the PyPIConGPU representation,
which you can dump to machine-readable `JSON <https://json.org/>`__.
Individual elements such as ``Species`` can be serialized and recovered
the same way.
Computed (read-only) fields must be excluded when dumping
(``exclude_computed_fields=True``),
and validators run again on ``model_validate``;
making the plain dump round-trip without that opt-out is tracked in
https://github.com/chillenzer-agents/picongpu/issues/65:

.. literalinclude:: ../snippets/defining_simulation/serialize_simulation.py
   :language: python
   :start-after: BEGIN-SERIALIZE-SIMULATION
   :end-before: END-SERIALIZE-SIMULATION

.. _serialization_metadata:

Metadata in a setup directory
-----------------------------

Generating the input files writes machine-readable descriptions
into the ``metadata/`` directory of the setup:

* ``metadata/pypicongpu_runner.json``:
  the state of the runner,
  including the complete PyPIConGPU representation of the simulation
  (the same JSON as ``get_as_pypicongpu().model_dump(mode="json")``).
* ``metadata/pypicongpu_rendering_context.json``:
  the full rendering context,
  i.e. all the values that went into the template rendering.
* ``metadata/rc_params.json``:
  the runtime configuration that was used.

In addition, the setup directory carries an
`RO-Crate <https://www.w3.org/TR/2021/REC-vc-r-crate-20211109/>`__
(version 1.2) as ``ro-crate-metadata.json``:
it describes the setup as a dataset,
with the ``workflow/workflow.cwl`` as its main entity,
the PIConGPU version as the software used,
and descriptions of the ``etc/``, ``include/``, ``metadata/`` and
``workflow/`` sub-datasets.
This makes the setup understandable to (and processable by) tools that
speak RO-Crate, independently of the PIConGPU Python package.

Reproducibility
---------------

To make a simulation reproducible, pin the PIConGPU version in the input
file's PEP 723 metadata:
replace ``@dev`` with a concrete ``@<commit hash>``.
The recorded metadata then documents exactly which version produced a run.

For workflow-level reproducibility, see also
:ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>`
on the workflow cache and the generated ``workflow/input.yaml``.
