/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
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

#include "picongpu/simulation/stage/CurrentDeposition.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldJ.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>


namespace picongpu::simulation::stage
{
    namespace detail
    {
        template<typename T_SpeciesType, typename T_Area>
        struct CurrentDeposition
        {
            using SpeciesType = T_SpeciesType;
            using FrameType = typename SpeciesType::FrameType;

            HINLINE void operator()(const uint32_t currentStep, FieldJ& fieldJ, pmacc::DataConnector& dc) const
            {
                auto species = dc.get<SpeciesType>(FrameType::getName());
                fieldJ.computeCurrent<T_Area::value, SpeciesType>(*species, currentStep);
            }
        };

    } // namespace detail

    void CurrentDeposition::operator()(uint32_t const step) const
    {
        using namespace pmacc;
        DataConnector& dc = Environment<>::get().DataConnector();
        auto& fieldJ = *dc.get<FieldJ>(FieldJ::getName());
        using SpeciesWithCurrentSolver =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, current<>>::type;
        meta::ForEach<
            SpeciesWithCurrentSolver,
            detail::CurrentDeposition<boost::mpl::_1, pmacc::mp_int<type::CORE + type::BORDER>>>
            depositCurrent;
        depositCurrent(step, fieldJ, dc);
    }
} // namespace picongpu::simulation::stage
