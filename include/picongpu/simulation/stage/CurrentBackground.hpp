/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/fields/background/cellwiseOperation.hpp"
#include "picongpu/fields/FieldJ.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            //! Functor for the stage of the PIC loop applying current background
            class CurrentBackground
            {
            public:
                /** Create a current background functor
                 *
                 * Having this in constructor is a temporary solution.
                 *
                 * @param cellDescription mapping for kernels
                 */
                CurrentBackground(MappingDesc const cellDescription) : cellDescription(cellDescription)
                {
                }

                /** Add the current background to the current density
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const step) const
                {
                    if(!FieldBackgroundJ::activated)
                        return;

                    using namespace pmacc;
                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto& fieldJ = *dc.get<FieldJ>(FieldJ::getName(), true);
                    using CurrentBackground = cellwiseOperation::CellwiseOperation<type::CORE + type::BORDER>;
                    CurrentBackground currentBackground(cellDescription);
                    currentBackground(&fieldJ, nvidia::functors::Add(), FieldBackgroundJ(fieldJ.getUnit()), step);
                    dc.releaseData(FieldJ::getName());
                }

            private:
                //! Mapping for kernels
                MappingDesc cellDescription;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
