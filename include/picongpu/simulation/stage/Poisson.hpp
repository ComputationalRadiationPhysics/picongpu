/* Copyright 2013-2024 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
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


 #include "picongpu/defines.hpp"
 #include "picongpu/fields/FieldJ.hpp"
 #include "picongpu/fields/FieldJ.kernel"
 #include "picongpu/fields/currentDeposition/Deposit.hpp"
 #include "picongpu/particles/filter/filter.hpp"
 #include "picongpu/particles/param.hpp"
 #include "picongpu/fields/FieldTmpOperations.hpp"
 
 #include <pmacc/Environment.hpp>
 #include <pmacc/dataManagement/DataConnector.hpp>
 #include <pmacc/meta/ForEach.hpp>
 #include <pmacc/particles/traits/FilterByFlag.hpp>
 #include <pmacc/type/Area.hpp>
 
 #include <cstdint>
 
 namespace picongpu
 {
     namespace simulation
     {
         namespace stage
         {
            //! Functor for the stage of the PIC loop performing charge deposition
            struct Poisson
            {
                /** Compute the current created by particles and add it to the current
                 *  density
                 *
                 * @param currentStep index of time iteration
                 */
                void operator()(uint32_t const currentStep) const;
            };

            namespace detail
            {
                 template<typename T_SpeciesType, typename T_Area>
                 struct Poisson
                 {
                     using SpeciesType = T_SpeciesType;
                     using FrameType = typename SpeciesType::FrameType;
 
                     /** Compute current density created by a species in an area */
                     HINLINE void operator()(uint32_t const currentStep, FieldJ& fieldJ, pmacc::DataConnector& dc) const
                     {
                     }
                 };

                 template<typename T_SpeciesType, typename T_Area>
                struct ComputeChargeDensity
                {
                    using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
                    static uint32_t const area = T_Area::value;

                    HINLINE void operator()(FieldTmp& fieldTmp, uint32_t const currentStep) const
                    {
                        DataConnector& dc = Environment<>::get().DataConnector();

                        /* load species without copying the particle data to the host */
                        auto speciesTmp = dc.get<SpeciesType>(SpeciesType::FrameType::getName());

                        /* run algorithm */
                        using ChargeDensitySolver = typename particles::particleToGrid::CreateFieldTmpOperation_t<
                            SpeciesType,
                            particles::particleToGrid::derivedAttributes::ChargeDensity>::Solver;

                        computeFieldTmpValue<area, ChargeDensitySolver>(fieldTmp, *speciesTmp, currentStep);
                    }
                };
                
             } // namespace detail
             template<typename T>
             using SpeciesEligibleForChargeDeposition =
                 typename particles::traits::SpeciesEligibleForSolver<T, simulation::stage::Poisson>::type;

             void Poisson::operator()(uint32_t const currentStep, MappingDesc *cellDescription) const
             {
                 using namespace pmacc;
                 constexpr uint fieldRhoSlot = 0;
                 DataConnector& dc = Environment<>::get().DataConnector();
                 auto& fieldRho = *dc.get<FieldTmp>(FieldTmp::getUniqueId(fieldRhoSlot));

                 fieldRho.getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));

                using EligibleSpecies = pmacc::mp_filter<SpeciesEligibleForChargeDeposition, VectorAllSpecies>;

                // todo: log species that are used / ignored in this plugin with INFO


                /* calculate and add the charge density values from all species in FieldTmp */
                meta::ForEach<
                    EligibleSpecies,
                    detail::ComputeChargeDensity<boost::mpl::_1, pmacc::mp_int<CORE + BORDER>>,
                    boost::mpl::_1>
                    computeChargeDensity;
                computeChargeDensity(fieldRho, currentStep);

                /* add results of all species that are still in GUARD to next GPUs BORDER */
                EventTask fieldTmpEvent = fieldRho.asyncCommunication(eventSystem::getTransactionEvent());
                eventSystem::setTransactionEvent(fieldTmpEvent);




                constexpr uint fieldVSlot = 1;
                auto& fieldV = *dc.get<FieldTmp>(FieldTmp::getUniqueId(fieldVSlot));
                fieldV.getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));

                BICGStab(fieldV, fieldRho, cellDescription);


             }
         } // namespace stage
     } // namespace simulation
     namespace particles
    {
        namespace traits
        {
            template<typename T_Species>
            struct SpeciesEligibleForSolver<T_Species, simulation::stage::Poisson>
            {
                using FrameType = typename T_Species::FrameType;

                // this plugin needs at least the weighting particle attribute
                using RequiredIdentifiers = MakeSeq_t<weighting>;

                using SpeciesHasIdentifiers =
                    typename pmacc::traits::HasIdentifiers<FrameType, RequiredIdentifiers>::type;

                // and also a charge ratio for a charge density
                using SpeciesHasFlags = typename pmacc::traits::HasFlag<FrameType, chargeRatio<>>::type;

                using type = pmacc::mp_and<SpeciesHasIdentifiers, SpeciesHasFlags>;
            };

        } // namespace traits
    } // namespace particles
 
} // namespace picongpu
 