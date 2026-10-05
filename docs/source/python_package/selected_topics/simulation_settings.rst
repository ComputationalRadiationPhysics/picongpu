.. _simulation_settings:

Simulation-Wide Settings
========================

Some options are configured on the ``Simulation`` itself
rather than on a grid, solver or species.

.. literalinclude:: ../snippets/selected_topics/simulation_settings.py
   :language: python
   :start-at: simulation = picmi.Simulation
   :end-before: simulation.write_input_file

Moving window
-------------

For simulations that follow a structure moving through the box
(e.g. a laser wakefield), the moving window slides the simulation window
along the positive ``y`` direction at the speed of light.
Configure it on the simulation:

* ``picongpu_moving_window_move_point``:
  the point a light ray reaches, measured from the left border in multiples
  of the simulation window size, until the window starts moving.
* ``picongpu_moving_window_stop_iteration``:
  the iteration at which to stop moving the window.

.. warning::

   While the moving window is active, one GPU row in ``y`` direction is
   reserved for initializing new space, which reduces the effective
   simulation window accordingly.
   The window must also be allowed to move within the grid:
   leave enough cells at the front and back of the box.

Normalization
-------------

PIConGPU normalizes the simulation to a reference density and a typical
particle count per cell:

* :attr:`~picongpu.picmi.simulation.Simulation.picongpu_base_density`:
  the reference density in m⁻³ (default ``1.0e25``).
  All species densities are normalized by it.
* ``picongpu_typical_ppc``:
  the typical number of macro-particles per cell used for code-unit
  normalization. If unset, the integer midpoint between the smallest and
  largest per-layout ``ppc`` of the initialized species is used.

Both are mostly relevant as numerical knobs and can usually be left at
their defaults.

Numerical precision
-------------------

:attr:`~picongpu.picmi.simulation.Simulation.picongpu_precision` selects the floating-point precision of the
simulation core:

* ``32``: single precision (the default),
* ``64``: double precision.

It controls the ``precisionPIConGPU`` namespace in the generated
``include/picongpu/param/precision.param``.
The special-operation namespaces (``sqrt``, ``exp``, trigonometric
functions) can be overridden individually via
``picongpu_precision_config`` (a ``picmi.PrecisionConfig``):
each of ``sqrt``, ``exp`` and ``trig`` takes ``"core"`` (the default,
which follows the core precision), ``32`` or ``64``.

Particle shape
--------------

:attr:`~picongpu.picmi.simulation.Simulation.particle_shape` sets the default
particle shape used for current/charge deposition, inherited by every
:class:`~picongpu.picmi.species.Species` that does not set its own
(see :ref:`Species, Distributions and Layouts <species>`).
It accepts the PICMI-standard shapes (``"NGP"``, ``"linear"``,
``"quadratic"``, ``"cubic"``) and PIConGPU ``"other:"`` extensions.
An explicit per-species ``particle_shape`` overrides it; if both are unset,
PIConGPU uses its native default ``"quadratic"`` (TSC).

Memory
------

``picongpu_memory_config`` (a ``picmi.MemoryConfig``) exposes the
low-level memory and exchange-buffer knobs rendered into
``include/picongpu/param/memory.param``:

* ``reserved_gpu_memory_size``: GPU-internal memory reserved by PIConGPU,
* ``bytes_exchange_x`` / ``bytes_exchange_y`` / ``bytes_exchange_z``:
  exchange-buffer sizes per direction,
* ``bytes_edges`` / ``bytes_corner``: exchange-buffer sizes for edge and
  corner communication,
* ``ref_local_dom_size`` / ``dir_scaling_factor``:
  reference local domain size and per-direction scaling for the exchange
  buffers,
* ``field_tmp_support_gather_communication``: whether temporary fields may
  gather neighbour information across devices.

All sizes are raw byte counts; write them with the byte-size constants from
``picongpu.picmi.constants`` (``KiB``, ``MiB``, ``GiB``, ...) for
readability, e.g. ``reserved_gpu_memory_size=350 * picmi.constants.MiB``.
(The ``super_cell_size`` is part of the grid, not the memory configuration.)

.. note::

   The compute backend is **not** part of the simulation script:
   it describes the architecture to compile for and belongs to the machine's
   runtime configuration as ``pic_backend`` (e.g. ``cuda:80``, ``hip:gfx90a``),
   see :ref:`Configuring Your Environment <python_package/foundations/configuring_environment:Configuring Your Environment>`.
   When a preset already provides a backend for the system,
   you usually do not need to set it yourself.

Wall-clock limit
----------------

:attr:`~picongpu.picmi.simulation.Simulation.picongpu_walltime` (a ``datetime.timedelta``) asks the scheduler to
stop the simulation once the given wall-clock time has passed
(the value must be positive).
On systems whose preset uses a batch scheduler this becomes the job's
time limit:

.. code-block:: python

   from datetime import timedelta

   sim = picmi.Simulation(..., picongpu_walltime=timedelta(hours=1))
