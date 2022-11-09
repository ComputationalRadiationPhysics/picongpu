.. _usage-particleBasedRadiationTransport:

Particle based Radiation Transport
==================================

.. sectionauthor:: Pawel Ordyna

1 Overview
----------

This page describes the already implemented parts of our particle based Monte-Carlo Radiation Transport implementation.
See :ref:`[1] <masterThesis-PawelOrdyna>` for more details.

2 Particle Injection
--------------------

It is possible to inject photon like macro particles at a simulation boundary for probing like setups.
The particles are spawned within the first layer of cells (line in 2D, plane in 3D) at a given simulation box side.
At the moment they can only be spawned with their momenta oriented along the direction normal to the boundary, so only probing along one of the simulation box axes is possible.

The beam particle spawning is similar to initial density initialization.
There is a fixed number of new particles in each boundary cell, either equally distributed within the initialization volume or randomly.
The particle weighting depends on the local beam density that is defined via an external beam transversal profile and temporal shape.
At the moment all particles are initialized with the same momentum (mono-energetic beam).
These initial attributes as well as additional ones are set with functors that can be configured in ``externalBeam.param``.
This param file also defines the beam profile and shape. The injection has to be also included in ``iterationStart.param``.

Example Configuration
^^^^^^^^^^^^^^^^^^^^^

Example ``externalBeam.param``:

.. code:: c++

   #pragma once

   #include "picongpu/param/grid.param"
   #include "picongpu/particles/externalBeam/beam/ProbingBeam.def"
   #include "picongpu/particles/externalBeam/beam/Side.hpp"
   #include "picongpu/particles/externalBeam/beam/beamProfiles/profiles.def"
   #include "picongpu/particles/externalBeam/beam/beamShapes/shapes.def"
   #include "picongpu/particles/externalBeam/functors.def"


   namespace picongpu::namespace particles::externalBeam
   {
        /* Choose from:
         *  - ZSide (probing along z)
         *  - YSide (probing along y)
         *  - XSide (probing along x)
         * - ZRSide (probing against z)
         * - YRSide (probing against y)
         * - XRSide (probing against x)
         */
        using ProbingSide = beam::ZSide;

        // Params for the Gaussian transversal beam profile (x,y are beam coordinates)
        PMACC_STRUCT(
            GaussParam,
            (PMACC_C_VALUE(float_X, sigmaX_SI, 5e-6))(
                PMACC_C_VALUE(float_X, sigmaY_SI, 5e-6)));
        /* Offset from the beam coordinate system default position (x,y) in beam coordinates.
         * As well as, the temporal delay.
         * The default position is the center of the initialization boundary.
         */
        PMACC_STRUCT(
            OffsetParam,
            (PMACC_C_VECTOR_DIM(float_X, DIM2, beamOffset_SI, 0.0, 0.0))(
                PMACC_C_VALUE(float_X, beamDelay_SI, 0.0)));

        // Constant temporal shape (square pulse)
        struct ConstShapeParam
        {
            static constexpr bool limitStart = true;
            static constexpr bool limitEnd = true;
            // does nothing since the limit is disabled
            static constexpr float_64 startTime_SI = 0.0;
            static constexpr float_64 endTime_SI =  10e-15;
        };

        using GaussianProfile = beam::beamProfiles::GaussianProfile<GaussParam>;
        using ConstShape = beam::beamShapes::ConstShape<ConstShapeParam>;

        using BeamProfile = GaussianProfile;
        using BeamShape = ConstShape;

        using Beam = beam::ProbingBeam<BeamProfile, BeamShape, ProbingSide, OffsetParam>;

        // Injection density defined by the beam and the maximum particle flux
        namespace density
        {
            struct ProbingBeamDensityParam
            {
            private:
                static constexpr float_64 cellVolumeSI
                    = SI::CELL_HEIGHT_SI * SI::CELL_WIDTH_SI * SI::CELL_DEPTH_SI;

            public:
                using ProbingBeam = Beam;
                //  100000 photons per full cell
                static constexpr float_64 photonFluxAtMaxBeamIntensity_SI{
                    100000.0 / cellVolumeSI * SI::SPEED_OF_LIGHT_SI};
            };
            using ProbingBeamDensity = ProbingBeamImpl<ProbingBeamDensityParam>;
        } // namespace density
        namespace startPosition
        {
            // Particle number and particle positioning within a cell:
            struct QuietProbingBeamParam
            {
                using Side = ProbingSide;
                /** Number of particles in each dimension initialized in a cell (in the beam coordinate
                 * system).
                 *
                 * Keep in mind that the particles are not spaced across the complete cell but rather a reduced
                 * cell. The cell dimensions along the beam x and y coordinates stay the same but along the
                 * beam z direction the cell depth is reduced to DELTA_T * SPED_OF_LIGHT.
                 *
                 * All 3 components need to be specified.  In the case of a 2 dimensional simulation, the
                 * component corresponding to the picongpu z direction will be discarded later.
                 */
                static constexpr float_X minWeighting = 0.001;
                using numParticlesPerDimension = mCT::Int<2, 2, 2>;
            };
            using QuietBeam = QuietProbingBeam<QuietProbingBeamParam>;
        } // namespace startPosition
        namespace momentum
        {
            // Initial particle momentum defined as the photon energy in Joule
            struct BeamMomentumParam
            {
                using Side = ProbingSide;
                static constexpr float_64 photonEnergySI = 6.0 * UNITCONV_keV_to_Joule;
            };
            using BeamMomentum = PhotonMomentum<BeamMomentumParam>;
        } // namespace momentum

        /* StartAttributes is an attribute initialization functor, the position and momentum functors
         * are always required. Any number of additional functors that are called in order can be added as optional
         * template arguments.
         */
            using BeamStartAttributes = StartAttributes<startPosition::QuietBeam, momentum::BeamMomentum>;
   } // namespace picongpu::namespace particles::externalBeam


Example ``iterationStart.param``:

.. code:: c++

    #pragma once

    #include "picongpu/particles/InitFunctors.hpp"


    namespace picongpu
    {
        /** IterationStartPipeline defines the functors called at each iteration start
         *
         * The functors will be called in the given order.
         *
         * The functors must be default-constructible and take the current time iteration as the only parameter.
         * These are the same requirements as for functors in particles::InitPipeline.
         */
        using IterationStartPipeline = pmacc::mp_list<particles::CreateDensity<
            particles::externalBeam::density::ProbingBeamDensity,
            particles::externalBeam::BeamStartAttributes,
            Photons>>;
    } // namespace picongpu

Supported functors
^^^^^^^^^^^^^^^^^^

probing beam configuration
""""""""""""""""""""""""""

.. doxygenstruct::  picongpu::particles::externalBeam::beam::ProbingBeam
   :project: PIConGPU

.. doxygenstruct::  picongpu::particles::externalBeam::beam::beamProfiles::ConstProfile
   :project: PIConGPU

.. doxygenstruct::  picongpu::particles::externalBeam::beam::beamProfiles::GaussianProfile
   :project: PIConGPU

.. doxygenstruct::  picongpu::particles::externalBeam::beam::beamShapes::ConstShape
   :project: PIConGPU

.. doxygenstruct::  picongpu::particles::externalBeam::beam::beamShapes::GaussianPulse
   :project: PIConGPU

.. doxygenstruct::  picongpu::particles::externalBeam::beam::beamShapes::LorentzPulse
   :project: PIConGPU

.. doxygenstruct::  picongpu::particles::externalBeam::beam::SqrtWrapper
   :project: PIConGPU

.. doxygenstruct::  picongpu::particles::externalBeam::density::ProbingBeamImpl
   :project: PIConGPU

.. doxygenstruct::  picongpu::particles::externalBeam::density::ProbingBeamImpl
   :project: PIConGPU
particle attributes
"""""""""""""""""""

.. doxygenstruct::  picongpu::particles::externalBeam::momentum::PhotonMomentum
   :project: PIConGPU

.. doxygenstruct::  picongpu::particles::externalBeam::phase::FromPhotonMomentum
   :project: PIConGPU

.. doxygenstruct::  picongpu::particles::externalBeam::phase::FromSpeciesWavelength
   :project: PIConGPU

.. doxygenstruct::  picongpu::particles::externalBeam::polarization::OnePolarization
   :project: PIConGPU

start position
""""""""""""""

.. doxygenstruct::  picongpu::particles::externalBeam::startPosition::QuietProbingBeam
   :project: PIConGPU

.. doxygenstruct::  picongpu::particles::externalBeam::startPosition::RandomProbingBeam
   :project: PIConGPU

References
----------

.. container:: references csl-bib-body
   :name: refs

    .. container:: csl-entry
       :name: masterThesis-PawelOrdyna

       [1]P. Ordyna, X-ray Radiation Transport in GPU Accelerated Particle in Cell Plasma Simulations, Zenodo, 2022. 10.5281/zenodo.7928423