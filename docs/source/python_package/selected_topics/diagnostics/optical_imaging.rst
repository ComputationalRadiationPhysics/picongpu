.. _optical-imaging:

Optical imaging (shadowgraphy)
==============================

The optical-imaging diagnostic time-integrates the Poynting vector in a fixed
plane of the simulation and applies Fourier-domain masks. It is the PICMI
frontend of the upstream ``shadowgraphy`` plugin; the plugin generalises beyond
shadowgraphy to *any* optical imaging resolved by the PIC method -- concrete
applications are combinations of the compile-time ``params`` values and the
three mask functions.

.. note::

   Requirements and limitations:

   * The plugin only works in **3D** simulations (the extraction plane is the
     ``z = slice_point * extent`` plane, so the probe pulse should propagate in
     ``z``); the frontend rejects a 2D grid.
   * It must be built with **FFTW3** and **openPMD** support
     (``PIC_ENABLE_FFTW3`` and ``ENABLE_OPENPMD``). FFTW3 is only an ``AUTO``
     CMake dependency, so a build without it will fail late.
   * The plugin is **not restart-safe**: the accumulated DFT state is not
     checkpointed (see `issue 132 <https://github.com/chillenzer-agents/picongpu/issues/132>`__).

Shadowgraphy preset
-------------------

``Shadowgraphy`` is a thin preset over ``OpticalImaging``: it supplies the
canonical Tukey windows in position and time, the band-pass and
numerical-aperture Fourier mask, and enables ``final_output`` so that a bare
call produces a shadowgram. The plugin is a multi-instance plugin, so several
diagnostics (e.g. two slices at different depths) can run side by side:

.. literalinclude:: ../../snippets/selected_topics/optical_imaging.py
   :language: python
   :start-at: BEGIN-OPTICAL-IMAGING-SHADOWGRAPHY
   :end-before: END-OPTICAL-IMAGING-SHADOWGRAPHY

Generic optical imaging
-----------------------

``OpticalImaging`` exposes the full surface and has **no defaults**: the three
mask functions must be supplied. They are ordinary Python callables of the
respective coordinates and must return `sympy <https://www.sympy.org/>`__
expressions (following the ``AnalyticDistribution`` pattern); they are rendered
into the C++ plugin by the frontend. A callable may reference the compile-time
``params::*`` constants and ``sim.*`` quantities by their C++ name, e.g. via
``sympy.Symbol("params::posWfSizeX")``:

.. literalinclude:: ../../snippets/selected_topics/optical_imaging_custom.py
   :language: python
   :start-at: BEGIN-OPTICAL-IMAGING-CUSTOM
   :end-before: END-OPTICAL-IMAGING-CUSTOM

The three callables are:

``position_wf(i, j, plugin_num_x, plugin_num_y)``
   Window function in the transverse position domain (applied while gathering
   the slice).

``time_wf(t, sim_num_t)``
   Window function in the time domain (applied during the time integration).

``mask_fourier(kx, ky, omega)``
   The general mask applied in Fourier space.

Runtime options
---------------

The following options are written per instance into ``N.cfg``
(``--shadowgraphy.*``):

``start``
   Step at which the time integration starts (default 0).

``duration``
   Length of the integration in simulation steps. Must be positive and is
   silently truncated to a multiple of ``t_res`` (validated by the frontend).

``file`` / ``ext``
   Output file prefix (default ``"shadowgram"``) and openPMD backend
   (default ``"bp5"``).

``slice_point``
   Position of the extraction plane as a ratio of the total ``z`` extent,
   ``0 <= slice_point < 1`` (default 0.5). Exactly ``1.0`` is outside the
   domain and rejected.

``focus_pos``
   Focus position of the Fourier propagator relative to the slice point, in SI
   metres (default 0.0).

``fourier_output`` / ``final_output``
   Optional openPMD outputs: the ``(x, y, omega)`` fields and the final
   shadowgram (which requires running the propagator). ``Shadowgraphy`` sets
   ``final_output=True``; a bare ``OpticalImaging`` does not.

.. note::

   The C++ plugin also registers a ``--shadowgraphy.intermediateOutput`` option,
   but never reads it. It is therefore deliberately not exposed by the Python
   frontend; do not expect ``(k_x, k_y, omega)`` output from it.

Compile-time parameters
-----------------------

The plugin constants (``t_res``, ``x_res``, ``y_res``,
``numerical_aperture``/``numerical_aperture_wf_size``,
``central_lambda``/``d_lambda``/``d_lambda_wf``, ``t_wf_buffer`` and
``pos_wf_size``) are compiled in. They are rendered into
``include/picongpu/param/shadowgraphy.param``, which shadows the packaged C++
default whenever a PICMI setup is generated. All ``OpticalImaging`` /
``Shadowgraphy`` instances of one simulation must agree on these compile-time
values (and on the three mask functions); only the runtime options may differ.
The mask functions and constants are the same knobs documented for the
:ref:`C++ plugin <usage-plugins-Shadowgraphy>`.

.. warning::

   The upstream runtime issue `#5706
   <https://github.com/ComputationalRadiationPhysics/picongpu/issues/5706>`__
   describes a CUDA-aware-MPI out-of-memory failure on JUPITER; it is an
   environment issue, not a configuration one, but is worth knowing about for
   multi-GPU production runs.
