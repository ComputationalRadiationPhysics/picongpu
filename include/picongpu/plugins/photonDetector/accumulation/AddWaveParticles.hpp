/* Copyright 2015-2021  Alexander Grund, Pawel Ordyna
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/PhotonFunctors.hpp"
#include "picongpu/plugins/photonDetector/DetectorParams.def"
#include "picongpu/plugins/photonDetector/PhotonDetectorUtilities.hpp"
#include "picongpu/plugins/photonDetector/accumulation/AddWaveParticles.def"

#include <pmacc/math/Complex.hpp>

namespace picongpu
{
    namespace plugins
    {
        namespace photonDetector
        {
            namespace accumulation
            {
                namespace detail
                {
                    /* This types are used to mock the constexpr if functionality from c++17.
                     * And they should be refactored out once static if becomes available. */

                    /** Get particle phase
                     *
                     * The particle phase can consist either of
                     *  1. A constant starting phase that can be different for each particle and a value that is common
                     *    for all particles in the species which is calculated from the simulation time and
                     *    the species wavelength flag.
                     *  2. An individual particle attribute phase that can be updated by e.g. the PhotonPhase pusher.
                     *
                     *  Approach 1 is used when the particle doesn't have the phase attribute. In that case the species
                     *  must have the wavelength flag and the startPhase attribute.
                     *  Approach 2 us used when the particle has the phase attribute.
                     *
                     * @tparam T_Species species type
                     * @tparam hasPhase true if the T_Species has the phase attribute
                     */
                    template<
                        typename T_Species,
                        bool hasPhase
                        = pmacc::traits::HasIdentifier<typename T_Species::FrameType, phase>::type::value>
                    struct GetPhase

                    {
                        //! Get the species wide contribution to particle phase at the give time step
                        HINLINE float_X getCurPhase(uint32_t const& currentStep) const
                        {
                            return 0.0_X;
                        }
                        //! Get the contribution to particle phase that can be different for each particle
                        template<typename T_Particle>
                        HDINLINE float_X getParticlePhase(T_Particle const& particle) const
                        {
                            return particle[phase_];
                        }
                    };


                    template<typename T_Species>
                    struct GetPhase<T_Species, false>
                    {
                        HINLINE float_X getCurPhase(uint32_t const& currentStep) const
                        {
                            return -1. * particles::GetPhaseByTimestep<T_Species>()(currentStep + 1) + 2. * PI;
                        }

                        template<typename T_Particle>
                        HDINLINE float_X getParticlePhase(T_Particle const& particle) const
                        {
                            return particle[startPhase_];
                        }
                    }; // namespace accumulation


                    /** Get angular frequency of a wave like particle
                     *
                     *  If the particle species has the wavelength flag, the angular frequency is identical for all
                     * particles and it is calculated from wavelength.
                     * In the other case, the angular frequency can vary between individual particles and is obtained
                     * from particle momentum.
                     *
                     * @tparam T_Species species type
                     * @tparam hasPhase true if T_Species has the wavelength flag.
                     */
                    template<
                        typename T_Species,
                        bool hasWavelength
                        = pmacc::traits::HasFlag<typename T_Species::FrameType, wavelength<>>::type::value>
                    struct GetAngFrequency
                    {
                        template<typename T_Particle>
                        HDINLINE float_X operator()(T_Particle const& particle) const
                        {
                            return particles::GetAngFrequency<T_Species>()();
                        }
                    };

                    template<typename T_Species>
                    struct GetAngFrequency<T_Species, false>
                    {
                        template<typename T_Particle>
                        HDINLINE float_X operator()(T_Particle const& particle) const
                        {
                            return particles::GetAngFrequency<T_Species>()(particle);
                        }
                    };
                } // namespace detail

                template<typename T_Species>
                HINLINE acc::AddWaveParticles<T_Species>::AddWaveParticles(
                    uint32_t currentStep,
                    const DetectorParams& detector,
                    DataSpace<simDim>& simSize)
                    : detector_m(detector)
                    , simSize_m(simSize)
                    , curPhase_m(detail::GetPhase<T_Species>().getCurPhase(currentStep))
                {
                }


                template<typename T_Species>
                template<typename T_Acc, typename T_DetectorBox, typename T_Particle>
                DINLINE void acc::AddWaveParticles<T_Species>::operator()(
                    T_Acc const& acc,
                    T_DetectorBox detectorBox,
                    const DataSpace<DIM2>& targetCellIdx,
                    const T_Particle& particle,
                    const DataSpace<simDim>& globalCellIdx) const
                {
                    using namespace pmacc;
                    // Get the dimensions of a picongpu cell in the detector coordinate system
                    const float_X CELL_LENGTH_DET_X_DIR = std::get<0>(getCellLengths(detector_m.detectorPlacement));
                    const float_X CELL_LENGTH_DET_Y_DIR = std::get<1>(getCellLengths(detector_m.detectorPlacement));
                    const float_X CELL_LENGTH_DET_Z_DIR = std::get<2>(getCellLengths(detector_m.detectorPlacement));
                    Type& oldVal = detectorBox(targetCellIdx);

                    /* Phase is: k*dn + w*t + phi_0 with dn...distance to detector, k...wavenumber, phi_0...start
                     * phase w*t is the same for all particles so we pre-calculate it (reduce to 2*PI) in high
                     * precision dn must be exact compared to lambda which is hard. We would also need to calculate
                     * dn to a fixed point for all photons in the detector-cell (e.g.) middle for proper
                     * interference In the end, only the phase difference matters. We take the ray from the cell
                     * (0,0) from the exiting plane as a reference and calculate the phase difference to this. It
                     * is the projection of the vector from the reference point to the particles position on the
                     * reference ray, whose vector is given by the particles direction (for large distances all
                     * rays to a given detector cell are parallel)
                     */

                    float_X phase;
                    // Combine the common and individual phase contributions.
                    // See detail::GetPhase for more detailed explanation.
                    phase = detail::GetPhase<T_Species>().getParticlePhase(particle) + curPhase_m;

                    if(phase > pmacc::math::Pi<float_X>::value)
                        phase -= pmacc::math::Pi<float_X>::doubleValue;

                    /* The projection is k * (dir * pos)/|dir| (dot product)
                     * We need the direction to a fixed point on the detector. As it is hard to calculate that
                     * exactly for the particle due to the large distance, we use the direction of the reference
                     * ray based on the observation that for "large" distances the angle is the same For better
                     * precision summands are reduced mod 2*PI
                     */
                    const pmacc::math::Vector<float_X, 2> detectorAngles
                        = precisionCast<float_X>(targetCellIdx - detector_m.size / 2) * detector_m.anglePerCell;
                    const pmacc::math::Vector<float_X, 3> dir(
                        math::tan<trigo_X>(detectorAngles.x()),
                        math::tan<trigo_X>(detectorAngles.y()),
                        1.0_X);
                    const float_X dirLen = math::abs(dir);

                    float_X omega;
                    omega = detail::GetAngFrequency<T_Species>()(particle);

                    const float_X k = omega / SPEED_OF_LIGHT;
                    // Reference is the middle of the end of the volume
                    DataSpace<simDim> globalCellOffset = globalCellIdx - simSize_m / 2;
                    globalCellOffset.z() = globalCellIdx.z() - simSize_m.z();
                    // Add the negated dot product (reduced by 2*PI), negated as the difference to the reference
                    // ray gets smaller with increasing index Add x,y parts first as those are much smaller than z
                    const float_X distDiffG = globalCellOffset.z() * CELL_LENGTH_DET_Z_DIR * dir.z()
                        + (globalCellOffset.x() * CELL_LENGTH_DET_X_DIR * dir.x()
                           + globalCellOffset.y() * CELL_LENGTH_DET_Y_DIR * dir.y());
                    phase += math::fmod(-distDiffG / dirLen * k, static_cast<float_X>(2 * PI));
                    // Now add the negated dot product for the remaining in-cell position
                    const float_X distDiffI = float_X(particle[position_].z()) * CELL_LENGTH_DET_Z_DIR * dir.z()
                        + (float_X(particle[position_].x()) * CELL_LENGTH_DET_X_DIR * dir.x()
                           + float_X(particle[position_].y()) * CELL_LENGTH_DET_Y_DIR * dir.y());
                    phase += math::fmod(-distDiffI / dirLen * k, static_cast<float_X>(2 * PI));

                    /*if(dir.z() > 1e-6)
                    {
                        printf("Dir: %g, %g, %g\n", dir.x(), dir.y(), dir.z());
                        printf("%i,%i -> %g+%g+%g+%g=%g -> %g\n", globalCellIdx.y(), globalCellIdx.z(),
                    particle[startPhase_], curPhase_, -distDiffG * k, -distDiffI * k, particle[startPhase_] +
                    curPhase_, phase+2*PI);
                    }*/

                    trigo_X sinPhase, cosPhase;
                    pmacc::math::sincos<trigo_X>(phase, sinPhase, cosPhase);
                    const trigo_X amplitude = particle[weighting_];
                    alpaka::atomicAdd(
                        acc,
                        &oldVal.get_real(),
                        FloatType(amplitude * cosPhase),
                        alpaka::hierarchy::Blocks{});
                    alpaka::atomicAdd(
                        acc,
                        &oldVal.get_imag(),
                        FloatType(amplitude * sinPhase),
                        alpaka::hierarchy::Blocks{});
                }

                template<typename T_Species>
                HINLINE std::vector<float_64> AddWaveParticles<T_Species>::getUnitDimension()
                {
                    /*
                     */
                    std::vector<float_64> unitDimension(7, 0.0);
                    unitDimension.at(SIBaseUnits::length) = 0.0;
                    unitDimension.at(SIBaseUnits::mass) = 0.0;
                    unitDimension.at(SIBaseUnits::time) = -0.0;
                    unitDimension.at(SIBaseUnits::electricCurrent) = 0.0;
                    return unitDimension;
                }
                template<typename T_Species>
                HDINLINE float_64 AddWaveParticles<T_Species>::getUnit()
                {
                    return 1.0;
                }

                template<typename T_Species>
                HINLINE std::string AddWaveParticles<T_Species>::getName()
                {
                    return "AddWaveParticles";
                }

                template<typename T_Species>
                HINLINE std::string AddWaveParticles<T_Species>::getOpenPMDMeshName()
                {
                    return "amplitude";
                }

                template<typename T_Species>
                HDINLINE acc::AddWaveParticles<T_Species> AddWaveParticles<T_Species>::operator()(
                    uint32_t currentStep,
                    const DetectorParams& detector,
                    DataSpace<simDim>& simSize) const
                {
                    return acc::AddWaveParticles<T_Species>(currentStep, detector, simSize);
                }
            } // namespace accumulation
        } // namespace photonDetector
    } // namespace plugins
} // namespace picongpu
