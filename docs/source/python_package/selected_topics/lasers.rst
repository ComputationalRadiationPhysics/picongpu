.. _lasers:

Lasers
======

Lasers are added to a simulation via the ``lasers`` parameter:

.. code-block:: python

   sim = picmi.Simulation(max_steps=..., solver=solver, lasers=[laser])

A Gaussian pulse specified by its normalized vector potential ``a0``:

.. literalinclude:: ../snippets/selected_topics/lasers.py
   :language: python
   :start-at: laser = picmi.GaussianLaser
   :end-before: grid = picmi.Cartesian3DGrid

A simulation can carry several lasers at once;
the :ref:`tutorial <python_package/tutorial:Tutorial: Setting up a simple LWFA>`
adds a Gaussian pulse to a full setup.

Analytic fields
---------------

:class:`~picongpu.picmi.lasers.GaussianLaser`,
:class:`~picongpu.picmi.lasers.PlaneWaveLaser`,
:class:`~picongpu.picmi.lasers.DispersivePulseLaser` and
:class:`~picongpu.picmi.lasers.TWTSLaser` can evaluate the analytic electric
field they describe, which is what PIConGPU injects for them.
:meth:`~picongpu.picmi.lasers.GaussianLaser.complex_amplitude` returns the
complex field amplitude, :meth:`~picongpu.picmi.lasers.GaussianLaser.envelope`
its absolute value, and
:meth:`~picongpu.picmi.lasers.GaussianLaser.E` the real vector field
(also available per component as ``Ex``/``Ey``/``Ez``).
:class:`~picongpu.picmi.lasers.DispersivePulseLaser` inherits and overrides
these: its :meth:`~picongpu.picmi.lasers.DispersivePulseLaser.complex_amplitude`
and :meth:`~picongpu.picmi.lasers.DispersivePulseLaser.E` additionally require
the simulation time step ``dt`` (the field is a finite discrete inverse Fourier
transform) and accept the translated initialization duration ``pulse_init``.
:class:`~picongpu.picmi.lasers.TWTSLaser` provides
:meth:`~picongpu.picmi.lasers.TWTSLaser.E` (per component ``Ex``/``Ey``/``Ez``)
and the corresponding :meth:`~picongpu.picmi.lasers.TWTSLaser.B`; both take the
domain center as the ``domain_center`` context argument, matching the core's
domain-center origin.
The coordinates ``x``, ``y`` and ``z`` may be numpy arrays of equal shape
(e.g. from ``numpy.meshgrid``); the vector field is returned
*component first*, i.e. with shape ``(3,) + x.shape``.
The convention matches the injected field: the on-axis, in-focus amplitude
of a :class:`~picongpu.picmi.lasers.GaussianLaser` is its ``E0``, the field
envelope decays as ``exp(-t^2 / duration^2)``, and the ``GaussianLaser``
``duration`` is the 1/e half-width of that field envelope.

The pulse is defined in the single, user-visible reference frame of the PICMI
input and the openPMD output: the pulse maximum (for a symmetric pulse, the
centroid) is at ``centroid_position`` at ``t = 0``.  ``E(x, y, z, t)`` is
evaluated in that frame, so it reproduces the field written to the result
files directly.  The Huygens-surface timing used internally by the PIConGPU
core is an implementation detail of the translation layer and never appears in
a user-facing formula.

.. literalinclude:: ../snippets/selected_topics/laser_fields.py
   :language: python
   :start-after: # BEGIN-LASER-FIELDS
   :end-before: # END-LASER-FIELDS

Common properties and constraints
---------------------------------

All lasers share a few properties:

* ``wavelength`` in metres,
* ``duration`` in seconds:
  for the standard Gaussian lasers (``GaussianLaser``, the dispersive pulse
  and ``TWTSLaser``) this is the 1/e half-width of the electric-field envelope
  (``E ~ exp(-t^2 / duration^2)``), i.e. the intensity (``E^2``) has the
  1-sigma width ``duration / 2``;
  ``PlaneWaveLaser`` instead takes ``duration`` directly
  as the 1-sigma width of the intensity profile,
* ``propagation_direction`` and ``polarization_direction``:
  normalized 3D vectors.
  The propagation direction must point *into* the simulation box;
  its components determine the entry faces
  (see `Propagation direction and entry faces`_ below).
* ``centroid_position``: the position of the pulse at time zero.
  It must be *outside* of the simulation box on every entry side
  (see `Propagation direction and entry faces`_ below),
  so that the pulse enters the box during the simulation.
* the field amplitude, given by exactly one of
  ``a0`` (the normalized vector potential) or
  ``E0`` (the peak electric field in V/m);
  the other is derived.

Propagation direction and entry faces
-------------------------------------

A laser enters the simulation box through the coordinate faces its
``propagation_direction`` *crosses*. A face is crossed iff the propagation
direction has a non-zero component pointing inward through that boundary:
for a positive ``d``-component this is the ``Min`` face of axis ``d`` (low
side), for a negative one the ``Max`` face (high side). For example
``[0, 1, 0]`` enters through ``YMin`` only (the conventional head-on case),
``[1, 0, 0]`` through ``XMin``, and ``[0, 0, -1]`` through ``ZMax``. By
default *all* crossed faces are used, so an obliquely incident pulse whose
direction has several non-zero components is injected through every one of
them: ``[0.5, 0, 0.866]`` crosses ``XMin`` **and** ``ZMin`` and is driven on
both faces with the one shared pulse profile. This is the physically correct
injection for an oblique wavefront, and it matches PIConGPU's ability to list
the same profile type under several face aliases.

Exactly the same rule validates ``centroid_position``: it must lie outside
the box on *every* entry side, i.e. for each crossed axis
``centroid[d] * propagation_direction[d] <= 0``.

The full box of Huygens surfaces (``picongpu_huygens_surface_positions``)
must be identical across all lasers, but each laser independently chooses
which of those surfaces it uses; injecting on a strict subset is allowed.
Use the PIConGPU extension keyword ``picongpu_entry_faces`` to give an
explicit per-laser face list, which overrides the derived all-crossed
default:

.. literalinclude:: ../snippets/selected_topics/laser_multi_face.py
   :language: python
   :start-after: # BEGIN-LASER-MULTI-FACE
   :end-before: # END-LASER-MULTI-FACE

A 2D (2D3V) simulation has no ``z`` coordinate and therefore no Z face to
inject through, so a direction with a ``z``-component (or an explicit
selection containing ``ZMin``/``ZMax``) is rejected there.

The example below places three lasers on three different faces — ``XMin``,
``YMax`` and ``ZMin`` — and checks the generated incident field:

.. literalinclude:: ../snippets/selected_topics/laser_entry_faces.py
   :language: python
   :start-after: # BEGIN-LASER-ENTRY-FACES
   :end-before: # END-LASER-ENTRY-FACES

:class:`~picongpu.picmi.lasers.TWTSLaser` is the exception: it keeps its
dedicated fixed placement (always ``YMin``, plus ``ZMin``/``ZMax`` chosen
by the sign of ``laserIncidenceAngle``) and its ``+y``-only validation,
regardless of the entry-face rule above.

Laser types
-----------

:class:`~picongpu.picmi.lasers.GaussianLaser`
   A Gaussian pulse.
   Structured beams can be described with the (matching-length) arrays
   ``picongpu_laguerre_modes`` and ``picongpu_laguerre_phases``.
   By default the polarization is linear;
   circular polarization is selected via
   ``picongpu_polarization_type=picmi.lasers.PolarizationType.CIRCULAR``.

:class:`~picongpu.picmi.lasers.DispersivePulseLaser`
   A Gaussian pulse with additional dispersion parameters:
   ``picongpu_spectral_support`` (width of the spectral support),
   ``picongpu_sd_si`` (spatial dispersion),
   ``picongpu_ad_si`` (angular dispersion),
   ``picongpu_gdd_si`` (group delay dispersion) and
   ``picongpu_tod_si`` (third-order dispersion).
   It does not support Laguerre modes.

:class:`~picongpu.picmi.lasers.TWTSLaser`
   An obliquely incident, pulse-front-tilted Gaussian pulse
   for traveling-wave Thomson-scattering setups;
   ``laserIncidenceAngle`` and ``polarizationAngle`` parameterize
   the incidence relative to the ``y`` axis.
   Its placement is fixed to the ``YMin``/``ZMin``/``ZMax`` faces
   (it always enters through ``YMin``); ``propagation_direction``
   must still point into the box (positive ``y`` component).
   Its focus is set by ``focal_position`` (the ``x`` coordinate is
   fixed to the domain center), and the analytic
   :meth:`~picongpu.picmi.lasers.TWTSLaser.E` /
   :meth:`~picongpu.picmi.lasers.TWTSLaser.B` additionally take the
   domain center as context (the C++ origin is the domain center).

:class:`~picongpu.picmi.lasers.PlaneWaveLaser`
   A plane wave with a temporal shape:
   a Gaussian ramp followed by a plateau of length
   ``picongpu_plateau_duration`` and a Gaussian fall-off.
   The focus is fixed at the origin
   (``focal_position`` and ``laser_nofocus_constant_si``
   are supplied by the frontend).

:class:`~picongpu.picmi.lasers.FromOpenPMDPulseLaser`
   A pulse imported from an `openPMD <https://www.openpmd.org/>`__ file
   (``file_path``, ``iteration``, ``dataset_name``, ...),
   for initial conditions that are too complex to describe analytically.

.. note::

   The PICMI-standard laser *injection method* is not supported:
   the injection method must be left at ``None``.

   ``picongpu_huygens_surface_positions`` optionally sets, per axis, the
   distance (in cells) from the global domain boundary at which the laser
   is injected; the distance must be at least the absorbing boundary size.
   The default is ``[[16, -16], [16, -16], [16, -16]]`` and rarely needs
   changing.
