.. _usage-plugins-xrayScattering:

xrayScattering
--------------

This plugin calculates Small Angle X-ray Scattering (SAXS) patterns from electron density.
( Using a density `FieldTmp` as an intermediate step and not directly the macro particle distribution. )
This is a species specific plugin and it has to be run separately for each scattering species.
Since the plugin output is the scattered complex amplitude, contributions from different species can be coherently summed later on. 

.. math::

   \Phi({\vec q}) &= \frac{r_e}{d}  \int_{t} \mathrm{d}t \int_{V} \mathrm{d}V \phi({\vec r}, t) n({\vec r}, t) \\
   I &= \left| \Phi \right|^2


============================== ================================================================================
Variable                       Meaning
============================== ================================================================================
:math:`\Phi`                   Scattered amplitude
:math:`\vec q`                  Scattering vector with :math:`|{\vec q}| = \frac{4 \pi \sin \theta}{\lambda}`
:math:`\theta`                 Scattering angle. :math:`2\theta` is the angle between the incoming and the scattered k-vectors.
:math:`\lambda`                Probing beam wavelength
:math:`n`                      Electron density
:math:`\phi`                   Incoming wave amplitude
:math:`I`                      Scattering intensity
:math:`d`                      Screen distance
:math:`r_e`                    Classical electron radius

============================== ================================================================================


For the free electrons, the density :math:`n` is just their number density, for ions it is the bound electrons density of the species.
This plugin will automatically switch to bound electrons density for species having the `boundElectrons` property.

The volume integral is realized by a discrete sum over the simulation cells and the temporal integration reduces to accumulating the amplitude over simulation time steps.

.. note::
    This calculation is based on the kinematic model of scattering. Multiple scattering CAN NOT be handled in this model.

.param file
^^^^^^^^^^^

The `xrayScattering.param` file sets the x-ray beam alignment as well as its temporal and transverse envelope.

.. note::
    At the moment the translation (to the side center + offset) is not working correctly.
    For that reason, the envelopes and the offset can't be set in the ``.param`` file yet.
    The probe is always a plane wave.
    Beam rotation works.

The alignment settings define a beam coordinate system with :math:`\hat{z}  = \hat{k}` and :math:`\hat{x}`, :math:`\hat{y}` perpendicular to the x-ray propagation direction.
It is always a right-hand system. It is oriented in such way that for propagation parallel to the PIC x- or y-axis (`Side`: `X`, `XR`, `Y` or `YR`) :math:`\hat{x}_{\text{beam}} = - \hat{z}_{\text{PIC}}` holds and if :math:`{\vec k }` is parallel to  the PIC z-axis (`Side`: `Z` or `ZR`),  :math:`\hat{x}_{\text{beam}} = - \hat{y}_{\text{PIC}}` holds.
The orientation can be then fine adjusted with the `RotationParam` setting.
.. TODO: Figures showing the beam coordinate system orientation in the PIC system.

.. TODO: Add other parameters after the coordinate transform has been fixed and the settings have been moved back to the .param file.

=================  ===============================================================================================================================
  Setting                      Description 
=================  ===============================================================================================================================
``ProbingSide``    The side from which the x-ray is propagated.
                   Set `X`, `Y` or `Z` for propagation along one of the PIC coordinate system axes;
                   `XR`, `YR` or `ZR` for propagation in an opposite direction.

``RotationParam``  Rotation of the beam axis, :math:`z_{\text{beam}}`, from the default orientation ( perpendicular the the simulation box side ).
                   Set the beam yaw and pitch angles in radians.
=================  ===============================================================================================================================

.. TODO: Add BEAM_OFFSET in between after the coordinate transform has been fixed.

The coordinate transfer from the PIC system to the beam system is performed in the following order:
rotation to one of the default orientations (``ProbingSide`` setting), additional rotation (``RotationParam`` ). This has to be taken into account when defining the experimental setup.


.cfg file
^^^^^^^^^

For a specific (charged) species ``<species>`` e.g. ``e``, the scattering can be computed by the following commands.

============================================ ============================================================================================================================================
Command line option                          Description
============================================ ============================================================================================================================================
``--<species>_xrayScattering.period``        Period at which the plugin is enabled (PIC period syntax). Only the intensity from this steps is accumulated.
                                             Default is `0`, which means that the scattering intensity in never calculated and therefor off

``--<species>_xrayScattering.outputPeriod``  Period at which the accumulated amplitude is written to the output file (PIC period syntax). Usually set close to the x-ray coherence time.

``--<species>_xrayScattering.qx_max``        Upper bound of reciprocal space range in qx direction. The unit is :math:`Å^{-1}`. Default is `5`.

``--<species>_xrayScattering.qy_max``        Upper bound of reciprocal space range in qy direction. The unit is :math:`Å^{-1}` Default is `5`.

``--<species>_xrayScattering.qx_min``        Lower bound of reciprocal space range in qx direction. The unit is :math:`Å^{-1}` Default is `-5`.

``--<species>_xrayScattering.qy_min``        Lower bound of reciprocal space range in qy direction. The unit is :math:`Å^{-1}` Default is `-5`.

``--<species>_xrayScattering.n_qx``          Number of scattering vectors needed to be calculated in qx direction. Default is `100`,

``--<species>_xrayScattering.n_qy``          Number of scattering vectors needed to be calculated in qy direction. Default is '100'.

``--<species>_xrayScattering.file``          Output file name. Default is `<species>_xrayScatteringOutput`.

``--<species>_xrayScattering.ext``           `openPMD` filename extension. This controls the backend picked by the `openPMD` API. Default is `bp` for adios backend.

``--<species>_xrayScattering.compression``   Backend-specific `openPMD` compression method (e.g.) zlib.

``--<species>_xrayScattering.memoryLayout``  Possible values: `mirror` and `split`. Output can be mirrored on all Host+Device pairs or uniformly split, in chunks, over all nodes.
                                             Use split when the output array is too big to store the complete computed q-space on one device.
                                             For small output grids the `mirror` setting could turn out to be more efficient.
============================================ ============================================================================================================================================


Output
^^^^^^

``<species>_xrayScatteringOutput.<backend-specific extension>``

Output file in the `openPMD` standard. An example on how to access your data with the python reader:

.. code-block:: python

    from picongpu.plugins.data import XrayScatteringData

    simulation_path = '...' # dir containing simOutput, input, ..,
    # Read output from the 0th step, for electrons, hdf5 backend.
    data = XrayScatteringData( simulation_path, 'e', 'h5' )
    amplitude = saxsData.get(iteration=0) * saxsData.get_unit()
    del XrayScatteringData

When you don't want to use the python reader keep in mind that:
 * All iterations are saved in a single file
 * The mesh containing the output is called `'amplitude'`
 * This mesh has 2 components,  `'x'` is the real part and `'y'` is the imaginary part.

.. note::
    The amplitude is not zeroed on ``outputPeriod`` so one has to subtract the output from the iteration one period before and then calculate :math:`\left|\Phi\right|^2` and sum it with the intensities from other coherence periods.


References
^^^^^^^^^^

- [1] Kluge, T., Rödel, C., Rödel, M., Pelka, A., McBride, E. E., Fletcher, L. B., … Cowan, T. E. (2017). Nanometer-scale characterization of laser-driven compression, shocks, and phase transitions, by x-ray scattering using free electron lasers. Physics of Plasmas, 24(10). https://doi.org/10.1063/1.5008289
