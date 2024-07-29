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

#include "picongpu/simulation/stage/CurrentInterpolationAndAdditionToEMF.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/EMFieldBase.tpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/currentInterpolation/CurrentInterpolation.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>


namespace picongpu::simulation::stage
{
    void CurrentInterpolationAndAdditionToEMF::registerHelp(boost::program_options::options_description& desc)
    {
        static constexpr auto hasCurrentBackground = FieldBackgroundJ::activated;
        // Current backround and binomial interpolation are incompatible #4250
        auto const options = std::string{"[none"} + (hasCurrentBackground ? "" : ", binomial") + "]";
        desc.add_options()(
            "currentInterpolation",
            boost::program_options::value<std::string>(&kindName),
            (std::string{"Current interpolation kind "} + options + " default: " + kindName).c_str());
    }


    void CurrentInterpolationAndAdditionToEMF::operator()(uint32_t const step) const
    {
        using namespace pmacc;

        DataConnector& dc = Environment<>::get().DataConnector();
        auto fieldSolver = dc.get<fields::Solver>(fields::Solver::getName());

        using SpeciesWithCurrentSolver =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, current<>>::type;
        auto const numSpeciesWithCurrentSolver = pmacc::mp_size<SpeciesWithCurrentSolver>::value;
        auto const existsParticleCurrent = numSpeciesWithCurrentSolver > 0;
        if(existsParticleCurrent)
        {
            auto& fieldJ = *dc.get<FieldJ>(FieldJ::getName());
            auto eRecvCurrent = fieldJ.asyncCommunication(eventSystem::getTransactionEvent());
            auto& interpolation = fields::currentInterpolation::CurrentInterpolation::get();
            auto const currentRecvLower = interpolation.getLowerMargin();
            auto const currentRecvUpper = interpolation.getUpperMargin();

            /* without interpolation, we do not need to access the FieldJ GUARD
             * and can therefore overlap communication of GUARD->(ADD)BORDER & computation of CORE
             */
            if(currentRecvLower == DataSpace<simDim>::create(0) && currentRecvUpper == DataSpace<simDim>::create(0))
            {
                fieldSolver->addCurrent<type::CORE>();
                eventSystem::setTransactionEvent(eRecvCurrent);
                fieldSolver->addCurrent<type::BORDER>();
            }
            else
            {
                /* in case we perform a current interpolation/filter, we need
                 * to access the BORDER area from the CORE (and the GUARD area
                 * from the BORDER)
                 * `fieldJ->asyncCommunication` first adds the neighbors' values
                 * to BORDER (send) and then updates the GUARD (receive)
                 * \todo split the last `receive` part in a separate method to
                 *       allow already a computation of CORE */
                eventSystem::setTransactionEvent(eRecvCurrent);
                fieldSolver->addCurrent<type::CORE + type::BORDER>();
            }
        }
        else
        {
            //! Whether current background is activated
            static constexpr auto hasCurrentBackground = FieldBackgroundJ::activated;
            /* With no current from macroparticles, there is no need for communication.
             * However we may still have J from the background (if it is activated) in CORE and BORDER.
             */
            if(hasCurrentBackground)
            {
                fieldSolver->addCurrent<type::CORE + type::BORDER>();
            }
        }
    }

    void CurrentInterpolationAndAdditionToEMF::init()
    {
        using namespace fields::currentInterpolation;
        auto& interpolation = CurrentInterpolation::get();
        // So far there are only two kinds and so names are hardcoded
        if(kindName == "none")
            interpolation.kind = CurrentInterpolation::Kind::None;
        else if(kindName == "binomial")
        {
            //! Whether current background is activated
            static constexpr auto hasCurrentBackground = FieldBackgroundJ::activated;
            // Current backround and binomial interpolation are incompatible #4250
            if(hasCurrentBackground)
                throw std::runtime_error(
                    "With current background enabled, only None current interpolation is allowed");
            interpolation.kind = CurrentInterpolation::Kind::Binomial;
        }
        else
            throw std::runtime_error("Unsupported current interpolation type");
    }
} // namespace picongpu::simulation::stage
