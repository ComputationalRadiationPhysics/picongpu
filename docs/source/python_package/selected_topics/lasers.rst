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

Common properties and constraints
---------------------------------

All lasers share a few properties:

* ``wavelength`` in metres,
* ``duration`` in seconds
  (the 1-sigma width of the intensity profile),
* ``propagation_direction`` and ``polarization_direction``:
  normalized 3D vectors.
  The propagation direction must point *into* the simulation box,
  i.e. have a positive ``y`` component.
* ``centroid_position``: the position of the pulse at time zero.
  It must be *outside* of the simulation box
  (``centroid_position[1] <= 0``),
  so that the pulse enters the box during the simulation.
* the field amplitude, given by exactly one of
  ``a0`` (the normalized vector potential) or
  ``E0`` (the peak electric field in V/m);
  the other is derived.

Laser types
-----------

``GaussianLaser``
   A Gaussian pulse.
   Structured beams can be described with the (matching-length) arrays
   ``picongpu_laguerre_modes`` and ``picongpu_laguerre_phases``.
   By default the polarization is linear;
   circular polarization is selected via
   ``picongpu_polarization_type=picmi.lasers.PolarizationType.CIRCULAR``.

``DispersivePulseLaser``
   A Gaussian pulse with additional dispersion parameters:
   ``picongpu_spectral_support`` (width of the spectral support),
   ``picongpu_sd_si`` (spatial dispersion),
   ``picongpu_ad_si`` (angular dispersion),
   ``picongpu_gdd_si`` (group delay dispersion) and
   ``picongpu_tod_si`` (third-order dispersion).
   It does not support Laguerre modes.

``TWTSLaser``
   An obliquely incident, pulse-front-tilted Gaussian pulse
   for traveling-wave Thomson-scattering setups;
   ``laserIncidenceAngle`` and ``polarizationAngle`` parameterize
   the incidence relative to the ``y`` axis.

``PlaneWaveLaser``
   A plane wave with a temporal shape.
   The focus is fixed at the origin
   (``focal_position`` and ``laser_nofocus_constant_si``
   are supplied by the frontend).

``FromOpenPMDPulseLaser``
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
