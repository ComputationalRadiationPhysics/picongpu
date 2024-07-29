/* Copyright 2021-2023 Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/algorithms/Velocity.hpp"
#include "picongpu/fields/absorber/Absorber.hpp"
#include "picongpu/fields/absorber/pml/Pml.kernel"
#include "picongpu/particles/InitFunctors.hpp"
#include "picongpu/particles/boundary/ApplyImpl.hpp"
#include "picongpu/particles/boundary/Kind.hpp"
#include "picongpu/particles/boundary/Parameters.hpp"
#include "picongpu/particles/boundary/Utility.hpp"
#include "picongpu/particles/functor/misc/Parametrized.hpp"
#include "picongpu/particles/manipulators/manipulators.hpp"
#include "picongpu/particles/manipulators/unary/FreeTotalCellOffset.hpp"
#include "picongpu/traits/attribute/DampedWeighting.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/boundary/Utility.hpp>
#include <pmacc/mappings/kernel/IntervalMapping.hpp>
#include <pmacc/traits/HasFlag.hpp>

#include <cstdint>


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            namespace detail
            {
                //! Functor to be applied to all particles in the active area
                struct AbsorbParticleIfOutside : public functor::misc::Parametrized<Parameters>
                {
                    //! Name, required to be wrapped later
                    static constexpr char const* name = "absorbParticleIfOutside";

                    /** Process the current particle located in the given cell
                     *
                     * @param offsetToTotalOrigin offset of particle cell in the total domain
                     * @param particle handle of particle to process (can be used to change attribute values)
                     */
                    template<typename T_Particle>
                    HDINLINE void operator()(DataSpace<simDim> const& offsetToTotalOrigin, T_Particle& particle)
                    {
                        if((offsetToTotalOrigin[m_parameters.axis] < m_parameters.beginInternalCellsTotal)
                           || (offsetToTotalOrigin[m_parameters.axis] >= m_parameters.endInternalCellsTotal))
                            particle[multiMask_] = 0;
                    }
                };

                /** Remove particles of the given species that are outer wrt the given boundary
                 *
                 * @tparam T_Species particle species type
                 *
                 * @param species particle species
                 * @param exchangeType exchange describing the active boundary
                 * @param currentStep current time iteration
                 */
                template<typename T_Species>
                HINLINE void removeOuterParticles(T_Species& species, uint32_t exchangeType, uint32_t currentStep)
                {
                    pmacc::DataSpace<simDim> beginInternalCellsTotal, endInternalCellsTotal;
                    getInternalCellsTotal(species, exchangeType, &beginInternalCellsTotal, &endInternalCellsTotal);
                    auto const axis = pmacc::boundary::getAxis(exchangeType);
                    AbsorbParticleIfOutside::parameters().axis = axis;
                    AbsorbParticleIfOutside::parameters().beginInternalCellsTotal = beginInternalCellsTotal[axis];
                    AbsorbParticleIfOutside::parameters().endInternalCellsTotal = endInternalCellsTotal[axis];
                    auto const mapperFactory = getMapperFactory(species, exchangeType);
                    using Manipulator = manipulators::unary::FreeTotalCellOffset<AbsorbParticleIfOutside>;
                    particles::manipulate<Manipulator, T_Species>(currentStep, mapperFactory);
                    species.fillGaps(mapperFactory);
                }

                //! Parameters to be passed to DampWeightsInPml functor
                struct DampWeightsInPmlParameters
                {
                    //! Axis of the active boundary
                    uint32_t axis;

                    //! Local PML parameters
                    fields::absorber::pml::LocalParameters localParameters;

                    //! From local domain start to total domain start, without guards
                    DataSpace<simDim> localToTotalDomainOffset;
                };

                /** Functor to be applied to all particles in the active area
                 *
                 * Follows approach in section II C of
                 * R. Lehe, A. Blelly, L. Giacomel, R. Jambunathan, J.-L. Vay
                 * Absorption of charged particles in Perfectly-Matched-Layers by optimal damping of the deposited
                 * current (2022) - version 2 from arXiv preprint at the time of our implementation.
                 * We further refer to this paper as [Lehe2022].
                 *
                 * Currently we do not store original (non-damped) weightings and permanently modify particle data.
                 * Instead, at each time step we calculate a damping multiplier for this time step and apply it.
                 * This means, using the paper notation, for us the integral is not from t_i to t but each time only
                 * over the current time step.
                 * As the particle's weight already has a combined effect from t_i to previous time step.
                 * If a particle never leaves PML after entering it, these formulations are equivalent.
                 *
                 * @TODO add an optional extra particle attribute to store additional weighting if a user chose that -
                 * then particles leaving PML would have their weights "restored", unlike in the current version.
                 */
                struct DampWeightsInPml : public functor::misc::Parametrized<DampWeightsInPmlParameters>
                {
                    //! Name, required to be wrapped later
                    static constexpr char const* name = "dampWeightsInPml";

                    /** Process the current particle located in the given cell
                     *
                     * @param offsetToTotalOrigin offset of particle cell in the total domain
                     * @param particle handle of particle to process (can be used to change attribute values)
                     */
                    template<typename T_Particle>
                    DINLINE void operator()(DataSpace<simDim> const& offsetToTotalOrigin, T_Particle& particle)
                    {
                        auto const axis = m_parameters.axis;
                        // Velocity half time step back relative to the current position
                        auto velocity = Velocity{};
                        auto const vel
                            = velocity(particle[momentum_], attribute::getMass(particle[weighting_], particle))[axis];

                        // Positions in local domain, in units of cells, without guards
                        auto const localCellIdx = offsetToTotalOrigin - m_parameters.localToTotalDomainOffset;
                        auto const positionCurrent
                            = (precisionCast<float_X>(localCellIdx) + particle[position_])[axis];
                        auto const positionPrevious = positionCurrent - vel * DELTA_T / cellSize[axis];

                        // Integral over the last time step as described in comment of this struct
                        auto const timeStepIntegral = fields::absorber::pml::getNormalizedSigmaIntegral(
                            positionPrevious,
                            positionCurrent,
                            DELTA_T,
                            m_parameters.localParameters,
                            axis);

                        // Adjust weighting according to [Lehe2022]
                        auto const weightingMultiplier = math::exp(-timeStepIntegral);
                        attribute::dampWeighting(particle, weightingMultiplier);
                    }
                };

                /** Damp weights of particles of the given species in PML, modifies weighting_ attribute
                 *
                 * @tparam T_Species particle species type
                 *
                 * @param species particle species
                 * @param exchangeType exchange describing the active boundary
                 * @param currentStep current time iteration
                 */
                template<typename T_Species>
                HINLINE void dampWeightsInPml(
                    T_Species& species,
                    uint32_t const exchangeType,
                    uint32_t const currentStep)
                {
                    // Do the procedure only for species with current deposition
                    using HasCurrentDeposition = typename HasFlag<typename T_Species::FrameType, current<>>::type;
                    if(!HasCurrentDeposition::value)
                        return;

                    // Do the procedure only for PML
                    auto const mappingDescription = species.getCellDescription();
                    auto& absorberImpl = fields::absorber::AbsorberImpl::getImpl(mappingDescription);
                    auto const kind = absorberImpl.getKind();
                    if(kind != fields::absorber::Absorber::Kind::Pml)
                        return;

                    // Our active area is between particle boundary offset and inner PML boundary
                    auto const axis = pmacc::boundary::getAxis(exchangeType);
                    auto const offsetCells = getOffsetCells(species, exchangeType);
                    auto const isMinSide = pmacc::boundary::isMinSide(exchangeType);
                    auto const& pmlImpl = absorberImpl.asPmlImpl();
                    auto const localParameters = pmlImpl.getLocalParameters(currentStep);
                    auto const pmlThickness = static_cast<uint32_t>(
                        isMinSide ? localParameters.negativeBorderSize[axis]
                                  : localParameters.positiveBorderSize[axis]);
                    if(offsetCells >= pmlThickness)
                        return;

                    // Create a mapping for the active area, take into account PML is never in guard area
                    auto const supercellSize = SuperCellSize::toRT()[axis];
                    auto const offsetFullSupercells = offsetCells / supercellSize;
                    auto const pmlSupercells = (pmlThickness + supercellSize - 1u) / supercellSize;
                    auto const guardSupercells = mappingDescription.getGuardingSuperCells();
                    auto beginSupercell = guardSupercells;
                    if(isMinSide)
                        beginSupercell[axis] = guardSupercells[axis] + offsetFullSupercells;
                    else
                        beginSupercell[axis]
                            = mappingDescription.getGridSuperCells()[axis] - guardSupercells[axis] - pmlSupercells;
                    auto numSupercells = mappingDescription.getGridSuperCells() - 2 * guardSupercells;
                    numSupercells[axis] = pmlSupercells - offsetFullSupercells;
                    auto const mapperFactory = pmacc::IntervalMapperFactory<simDim>{beginSupercell, numSupercells};

                    auto const& subGrid = Environment<simDim>::get().SubGrid();
                    auto const localToTotalDomainOffset
                        = subGrid.getGlobalDomain().offset + subGrid.getLocalDomain().offset;
                    DampWeightsInPml::parameters().axis = axis;
                    DampWeightsInPml::parameters().localParameters = localParameters;
                    DampWeightsInPml::parameters().localToTotalDomainOffset = localToTotalDomainOffset;
                    using Manipulator = manipulators::unary::FreeTotalCellOffset<DampWeightsInPml>;
                    particles::manipulate<Manipulator, T_Species>(currentStep, mapperFactory);
                }

            } // namespace detail

            //! Functor to apply absorbing boundary to particle species
            template<>
            struct ApplyImpl<Kind::Absorbing>
            {
                /** Apply absorbing boundary conditions along the given outer boundary
                 *
                 * @tparam T_Species particle species type
                 *
                 * @param species particle species
                 * @param exchangeType exchange describing the active boundary
                 * @param currentStep current time iteration
                 */
                template<typename T_Species>
                void operator()(T_Species& species, uint32_t const exchangeType, uint32_t const currentStep)
                {
                    detail::removeOuterParticles(species, exchangeType, currentStep);
                    detail::dampWeightsInPml(species, exchangeType, currentStep);
                }
            };

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
