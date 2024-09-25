.. _how_to_setup_ionization:

Ionization
==========
(See :ref:`usage-params` for how to configure a picongpu simulation in general)

.. note::
    The `default ionizer.param <https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/include/picongpu/param/ionizer.param>`_ and `default ionizationEnergies.param <https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/include/picongpu/param/ionizationEnergies.param>`_ includes all atomic parameters for some species.

To include charge-state-only simulations in your simulations you must:

1. Define the atomic parameters of all ionization ion species of your simulation in the :code:`ionizer.param`.

    1.) number of protons and neutrons

        .. code-block:: c++

            /* Example: nitrogen */
            namespace picongpu::ionization::atomicNumbers
            {
                struct Nitrogen
                {
                    static constexpr float_X numberOfProtons  = 7.0;
                    static constexpr float_X numberOfNeutrons = 7.0;
                };
            } // namespace picongpu::ionization::atomicNumbers

    2.) effective nuclear charge for each charge state

        .. code-block:: c++

            /* Example: nitrogen */
            namespace picongpu::ionization::effectiveNuclearCharge
            {
                /// @attention insert the values in REVERSE order since the lowest shell corresponds to the last ionization process.
                PMACC_CONST_VECTOR(
                    float_X,
                    7, // number of charge states
                    Nitrogen, // name you may reference this by later, remember to prepend the namespace!
                    /* 2p^3 */
                    3.834,
                    3.834,
                    3.834,
                    /* 2s^2 */
                    3.874,
                    3.874,
                    /* 1s^2 */
                    6.665,
                    6.665);
            } // namespace picongpu::ionization::effectiveNuclearCharge

        .. note::
            see `wikipedia <https://en.wikipedia.org/wiki/Effective_nuclear_charge>`_ for values or refer directly to the calculations by Clementi and Raimondi, [1]_ and [2]_.

    .. attention::
        In addition this file must also contain the Thomas-Fermi ionization parameters and cutoff settings.
        We suggest to copy them from the default :code:`ionizer.param` and **not** adjust them.

2. Define the ionization energies of all charge states of all ionization ion species of your simulation in :code:`ionizationEnergies.param`.

    .. code-block:: c++

        namespace picongpu::ionization::energies::AU
        {
            /* example Nitrogen */

            /* ionization energy in eV */
            PMACC_CONST_VECTOR(
                float_X,
                7, // number charge states
                Nitrogen, // name you may reference this by later, remember to prepend the namespace and append _t!
                sim.si.conv().eV2auEnergy(14.53413),
                sim.si.conv().eV2auEnergy(29.60125),
                sim.si.conv().eV2auEnergy(47.4453),
                sim.si.conv().eV2auEnergy(77.4735),
                sim.si.conv().eV2auEnergy(97.89013),
                sim.si.conv().eV2auEnergy(552.06731),
                sim.si.conv().eV2auEnergy(667.04609));
        }; // namespace picongpu::ionization::energies::AU

    .. note::
         see `NIST <http://physics.nist.gov/PhysRefData/ASD/ionEnergy.html>`_ for ionization energies of the elements [3]_


3. Define mass and charge ratios of ions in the :code:`speciesDefinition.param`.

    .. code-block:: c++

        /* Example Nitrogen*/
        namespace picongpu
        {
            /* mass and charge ratios with respect to electrons */
            value_identifier(float_X, MassRatioNitrogen, 25514.325);
            value_identifier(float_X, ChargeRatioNitrogen, -7.0);
        } // namespace picongpu

4. Define at least one electron species in :code:`speciesDefinition.param`.

    .. code-block:: c++

        /* Example electron Species */
        namespace picongpu
        {
            using ParticleFlagsElectrons = MakeSeq_t<
                particlePusher<UsedParticlePusher>,
                shape<UsedParticleShape>,
                interpolation<UsedField2Particle>,
                current<UsedParticleCurrentSolver>,
                massRatio<MassRatioElectrons>,
                chargeRatio<ChargeRatioElectrons>>;

            using ParticleAttributesElectrons = MakeSeq_t<position<position_pic>, momentum, weighting>;

            using Electrons = Particles<PMACC_CSTRING("e"), ParticleFlagsElectrons, ParticleAttributesElectrons>;
        } // namespace picongpu


5. Define all (ion) macro particle species.

    The definition of an ionization species must include the particle attribute :code:`boundElectrons`, and the particle flags :code:`atomicNumbers`, :code:`ionizationEnergies`, :code:`effectiveNuclearCharge`, :code:`ionizers`, :code:`massRatio` and :code:`chargeRatio`.

    .. code-block:: c++

        namespace picongpu
        {
            #ifndef PARAM_IONIZATIONCURRENT
            #    define PARAM_IONIZATIONCURRENT None
            #endif

            /* Example Nitrogen */
            using ParticleFlagsNitrogen = MakeSeq_t<
                particlePusher<UsedParticlePusher>,
                shape<UsedParticleShape>,
                interpolation<UsedField2Particle>,
                current<UsedParticleCurrentSolver>,
                atomicNumbers<ionization::atomicNumbers::Nitrogen>                      // <-- from step 1
                effectiveNuclearCharge<ionization::effectiveNuclearCharge::Nitrogen_t>, // <-- from step 1
                ionizationEnergies<ionization::energies::AU::Nitrogen_t >               // <-- from step 2
                massRatio<MassRatioNitrogen>,                                           // <-- from step 3
                chargeRatio<ChargeRatioNitrogen>,                                       // <-- from step 3
                ionizers<MakeSeq_t<               // <-- comma separated list of all ionizers of this species with
                                                  // species of electrons macro particles to be created upon ionization
                    particles::ionization::BSIEffectiveZ<Electrons, particles::ionization::current::PARAM_IONIZATIONCURRENT>,
                    particles::ionization::ADKLinPol<Electrons, particles::ionization::current::PARAM_IONIZATIONCURRENT>,
                    particles::ionization::ThomasFermi<Electrons>>>>;

            using ParticleAttributesNitrogen = MakeSeq_t<position<position_pic>, momentum, weighting, boundElectrons>;
            using Nitrogen = Particles<PMACC_CSTRING("N"), ParticleFlagsNitrogen, ParticleAttributesNitrogen>;
        } // namespace picongpu

    .. note::
        Remember that you can define multiple electron species and associate them with your different ion species to track the ionization processes separately!

6. Create ion macro particles in the simulation and initialize them in :code:`speciesInitialization.param`.

    .. code-block:: c++

        namespace picongpu::particles
        {
            /* create nitrogen charge state 1 and electrons to neutralize the simulation */
            using InitPipeline = pmacc::mp_list<
                CreateDensity< ... some densityProfile... , startPosition::Random, Nitrogen>,
                ManipulateDerive<manipulators::binary::DensityWeighting, Nitrogen, Electrons>,
                Manipulate<manipulators::unary::ChargeState<1u>;, Nitrogen>>;
        } // namespace picongpu::particles

    .. attention::
        Ensure the simulation is initialized as charge neutral whenever the respective species should move, to avoid creating unintended ghost background charge densities.

    .. note::
    For an example of a picongpu setup with ionization see the `FoilLTC example <https://github.com/ComputationalRadiationPhysics/picongpu/tree/dev/share/picongpu/examples/FoilLCT>`_.

.. [1]
    Clementi, E.; Raimondi, D. L. (1963)
    "Atomic Screening Constants from SCF Functions"
    J. Chem. Phys. 38 (11): 2686-2689.
    https://dx.doi.org/10.1063/1.1733573
.. [2]
    Clementi, E.; Raimondi, D. L.; Reinhardt, W. P. (1967)
    "Atomic Screening Constants from SCF Functions. II. Atoms with 37 to 86 Electrons"
    Journal of Chemical Physics. 47: 1300-1307
    https://dx.doi.org/10.1063/1.1712084
.. [3]
    Kramida, A., Ralchenko, Yu., Reader, J., and NIST ASD Team (2014)
    NIST Atomic Spectra Database (ver. 5.2), [Online]
    https://dx.doi.org/10.18434/T4W30F [2017, February 8]
    National Institute of Standards and Technology, Gaithersburg, MD
    also available via: http://physics.nist.gov/asd
