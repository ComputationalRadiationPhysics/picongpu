/**
 * Copyright 2016-2017 Heiko Burau
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "simulation_defines.hpp"
#include "ScaledSpectrum.hpp"
#include "PhotonEmissionAngle.hpp"
#include "fields/FieldTmp.hpp"

#include "random/methods/XorMin.hpp"
#include "random/distributions/Uniform.hpp"
#include "random/RNGProvider.hpp"

#include "traits/Resolve.hpp"


namespace picongpu
{
namespace particles
{
namespace bremsstrahlung
{

/** Handling of the Bremsstrahlung effect.
 *
 * Here the screened Bethe-Heitler cross section is used. See e.g.:
 * Salvat, F., et al. "Monte Carlo simulation of bremsstrahlung emission by electrons."
 * Radiation Physics and Chemistry 75.10 (2006): 1201-1219.
 *
 * The numerics separates the energy spectrum into two parts. In the low-energy part
 * photon emission is neglected and a drag force is applied to the electrons. In the high-energy part
 * photons are created in addition to the drag force.
 *
 * Electron deflection is treated as screened Rutherford scattering, see e.g. Jackson, chap. 13.5
 *
 * The photon emission angle is taken from the Lorentz-boosted dipole radiation formula,
 * see e.g. Jackson, chap. 15.2
 *
 * \tparam T_ElectronSpecies
 * \tparam T_PhotonSpecies
 */
template<typename T_IonSpecies, typename T_ElectronSpecies, typename T_PhotonSpecies>
struct Bremsstrahlung
{
    typedef T_IonSpecies IonSpecies;
    typedef T_ElectronSpecies ElectronSpecies;
    typedef T_PhotonSpecies PhotonSpecies;

    typedef typename ElectronSpecies::FrameType FrameType;

    /* specify field to particle interpolation scheme */
    typedef typename PMacc::traits::Resolve<
        typename GetFlagType<FrameType,interpolation<> >::type
        >::type Field2ParticleInterpolation;

    /* margins around the supercell for the interpolation of the field on the cells */
    typedef typename GetMargin<Field2ParticleInterpolation>::LowerMargin LowerMargin;
    typedef typename GetMargin<Field2ParticleInterpolation>::UpperMargin UpperMargin;

    /* relevant area of a block */
    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        LowerMargin,
        UpperMargin
        > BlockArea;

    BlockArea BlockDescription;

    typedef MappingDesc::SuperCellSize TVec;

    typedef FieldTmp::ValueType ValueTypeIonDensity;

private:
    /* global memory ion density field device databoxes */
    PMACC_ALIGN(ionDensityBox, FieldTmp::DataBoxType);
    /* shared memory ion density device databoxes */
    PMACC_ALIGN(cachedIonDensity, DataBox<SharedBox<ValueTypeIonDensity, typename BlockArea::FullSuperCellSize, 0> >);

    PMACC_ALIGN(scaledSpectrumFunctor, ScaledSpectrum::LookupTableFunctor);
    PMACC_ALIGN(stoppingPowerFunctor, ScaledSpectrum::LookupTableFunctor);
    PMACC_ALIGN(getPhotonAngleFunctor, GetPhotonAngle::GetPhotonAngleFunctor);

    PMACC_ALIGN(photonMom, float3_X);

    /* random number generator */
    typedef PMacc::random::RNGProvider<simDim, PMacc::random::methods::XorMin> RNGFactory;
    typedef PMacc::random::distributions::Uniform<float> Distribution;
    typedef typename RNGFactory::GetRandomType<Distribution>::type RandomGen;
    RandomGen randomGen;

public:
    /* host constructor initializing member */
    Bremsstrahlung(
        const ScaledSpectrum::LookupTableFunctor& scaledSpectrumFunctor,
        const ScaledSpectrum::LookupTableFunctor& stoppingPowerFunctor,
        const GetPhotonAngle::GetPhotonAngleFunctor& getPhotonAngleFunctor,
        const uint32_t currentStep);

    /** Initialization function on device
     *
     * \brief Cache ion density field on device
     *         and initialize possible prerequisites, like e.g. random number generator.
     *
     * This function will be called inline on the device which must happen BEFORE threads diverge
     * during loop execution. The reason for this is the `__syncthreads()` call which is necessary after
     * initializing the ion density field in shared memory.
     */
    DINLINE void init(const DataSpace<simDim>& blockCell, const int& linearThreadIdx, const DataSpace<simDim>& localCellOffset);

    /** Rotates a vector to a given polar angle and a random azimuthal angle.
     *
     * @param vec vector to be rotated
     * @param theta polar angle
     * @return rotated vector
     */
    DINLINE float3_X scatterByTheta(const float3_X vec, const float_X theta);

    /** Return the number of target particles to be created from each source particle.
     *
     * Called for each frame of the source species.
     *
     * @param sourceFrame Frame of the source species
     * @param localIdx Index of the source particle within frame
     * @return number of particle to be created from each source particle
     */
    DINLINE unsigned int numNewParticles(FrameType& sourceFrame, int localIdx);

    /** Functor implementation.
     *
     * Called once for each single particle creation.
     *
     * \tparam Electron type of electron which creates the photon
     * \tparam Photon type of photon that is created
     */
    template<typename Electron, typename Photon>
    DINLINE void operator()(Electron& electron, Photon& photon);
};

} // namespace bremsstrahlung
} // namespace particles
} // namespace picongpu
