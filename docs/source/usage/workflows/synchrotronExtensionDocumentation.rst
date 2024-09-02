.. _synchrotronRadiation:

Synchrotron Radiation Extension
=================================================================

Introduction
------------

This documentation provides an overview of the Synchrotron Radiation Extension implemented in PIConGPU. Synchrotron radiation is a phenomenon where charged particles emit radiation due to acceleration by electromagnetic fields. This extension models the emission of photons from electrons under specific conditions, primarily in ultrastrong laser fields.

------------------------------

Synchrotron radiation occurs when charged particles moving at relativistic speeds are accelerated by magnetic and/or electric fields, causing them to emit energy in the form of electromagnetic radiation. This process is relevant in various fields, including astrophysics, particle accelerators, and plasma physics.

Model Used
----------

The model implemented in this extension is based on the assumptions and formulations described in Gonoskov et al., 2015 [Gonoskov2015]_. The implementation simulates one process as described in section B.1 "Photon emission and nonlinear Compton scattering". The specific algorithm is described in section G "Modified event generator". The key assumptions of the model include:

- **Insignificant Variation of Fields during one time step:** The electric (:math:`\vec E`) and magnetic (:math:`\vec B`) fields are assumed to vary insignificantly during one time step of the PIC simulation.
- **Ultrarelativistic Case:** The electrons are considered to be in an ultrarelativistic state, where the Lorentz factor :math:`\gamma \gg 1`.
- **Transverse Acceleration Dominance:**  emission of a particle is predominantly defined by the transverse acceleration (the longitudinal acceleration has :math:`\gamma^2` times less contribution to the emission intensity).
- **Effective Magnetic Field:** The emission properties can be approximated by considering an electron rotating in a constant uniform magnetic field (:math:`H_{\text{eff}}`), which is determined by the transverse acceleration. For this condition to be met the assumption :math:`\gamma \sim a_0 \gg 100` is considered. (:math:`a_0` is the normalized electric field strength of the laser field).
- **Forward Direction Emission:** The radiation emitted by the electrons is in the forward direction relative to their motion.

How and When to Use Synchrotron Radiation
-----------------------------------------

Synchrotron radiation can be used in simulations where the dynamics of high-energy particles in strong electromagnetic fields are of interest, such as in scenarios involving:

- High-intensity laser interactions with matter.
- Dynamics in astrophysical environments.
- Studies of radiation damping effects in particle accelerators.

Essentially, whenever the assumption :math:`\gamma \sim a_0 \gg 100` is valid.
This extension was primarily developed to study the effects of radiation on particle dynamics and energy distribution in plasmas under extreme field conditions.

Parameters Available to the User in syncrotron.param
-----------------------------------------------------

Param: ``ElectronRecoil``
~~~~~~~~~~~~~~~~~~~~~~~~~~	

- **Type:** `bool`
- **Default:** `false`
- **Description:** Enables or disables the electron recoil effect due to photon emission. When set to `true`, the momentum change of electrons due to emitted photons is considered in the simulation.

Param: ``minEnergy``
~~~~~~~~~~~~~~~~~~~~~~~~~~	 
- **Type:** `float_64`
- **Unit:** PIC units 
- **Default:** `HBAR / sim.pic.getDt()`
- **Description:** Sets the minimum energy threshold for photons to be considered in the simulation. This parameter helps in filtering out low-energy photons that may be already accounted for by the PIC fields. The default value is an approximation for the maximum photon energy that can be resolved by the field grid. This parameter is dependent on the internal unit system used in PIConGPU (Please use variables defined in `/picongpu/include/picongpu/unitless/*.unitless`).

Precomputation Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~
The algorithm precomputes necessary functions namely first and second synchrotron functions described in section E. `Spectrum of emission` of [Gonoskov2015]_.

`FirstSynchrotronFunctionParams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Description:** The function for precomputation of the first synchrotron function is in: ``picongpu/include/picongpu/simulation/stage/SynchrotronRadiation.hpp``

  - ``logEnd``: sets a cutoff for computing the Bessel function. After this point we assume the function is close enough to zero to be neglected. 
  - ``numberSamplePoints``: Number of sample points used in the numerical integration for the first synchrotron function.

`InterpolationParams`
~~~~~~~~~~~~~~~~~~~~~~

- **Description:** Parameters for setting up the interpolation table used in the simulation for efficient computation.

  - ``numberTableEntries``: Number of entries in the precomputed table for synchrotron function values.
  - ``minZqExponent``: connected to the range of :math:`z_q` (eq.21 [1]_) values. The minimum :math:`z_q` value is :math:`\log_2(\text{minZqExponent})`
  - ``maxZqExponent``: respectively as ``minZqExponent``

Param: ``supressRequirementWarning``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	 
The algorithm implements checking for requirements described in [1]_ in sec. H. "Adaptive event generator". If the requirements are not fulfilled that means that that the propability of generating a photon is high for given `dt` (higher than 10%) this means that photons are generated possibly every timestep, which causes numerical artefacts, and the radiation is underestimeted. In that case the timestep should be reduced. A script is provided for calculating maximum timestep for given parameters. The script can be found in `lib/python/synchrotronRadiationExtension/synchrotronRequirements.py`.

- **Type:** `bool`
- **Default:** `false`
- **Description:** Suppresses the warning related to the requirements of the synchrotron radiation model. If set to `true`, it avoids the runtime checks, potentially increasing simulation speed at the risk of accuracy.

The extensions notifies the user of failed requirements by printing out:
 - first time - a warning
 - consecutive - a dot (".")

Closing Notes
-------------

The Synchrotron Radiation extension for PIConGPU is a tool for simulating high-energy radiation from physics scenarios involving ultrarelativistic, charged particles in strong electromagnetic fields. Users are encouraged to understand the underlying assumptions and limitations of the model used in this extension to effectively interpret the results of their simulations. For detailed theoretical background and further reading, refer to the cited literature by Gonoskov et al., 2015.

References
----------

.. [Gonoskov2015]
        A. Gonoskov et. al. 
        *Extended particle-in-cell schemes for physics in ultrastrong laser fields: Review and developments*
        PHYSICAL REVIEW E (2015)
        https://doi.org/10.1103/PhysRevE.92.023305
