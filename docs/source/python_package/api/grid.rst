Grid
====

.. automodule:: picongpu.picmi.grid
   :members:
   :undoc-members:
   :show-inheritance:

Absorbing field reference
-------------------------

The absorbing field (PML / exponential) is configured on the grid:

* ``pml_cells`` (standard PICMI): per-axis, symmetric absorber depth in cells.
* ``picongpu_absorber_kind``: the absorber profile, ``"pml"`` (default) or
  ``"exponential"``; rendered as the runtime option ``--fieldAbsorber``.
* ``picongpu_pml_cells``: per-axis, per-direction depth
  ``[[negative, positive], ...]`` (full ``NUM_CELLS[3][2]``), mutually exclusive
  with ``pml_cells``.
* ``picongpu_exponential_strength``: per-axis, per-direction
  ``exponential::STRENGTH`` ``[[negative, positive], ...]``.

See :ref:`grids_field_absorber` for usage and validation.
