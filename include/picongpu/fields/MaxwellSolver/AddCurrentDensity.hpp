/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch, Benjamin Worpitz, Sergei Bastrakov
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

#include "picongpu/fields/Fields.hpp"
#include "picongpu/fields/MaxwellSolver/AddCurrentDensity.kernel"
#include "picongpu/fields/currentInterpolation/CurrentInterpolation.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>

#include <cstdint>


namespace picongpu::fields::maxwellSolver
{
    /** Functor to smooth current density and add it to the electric field
     *
     * @tparam T_area area to operate on
     */
    template<uint32_t T_area>
    struct AddCurrentDensity
    {
        /** Create functor instance
         *
         * @param cellDescription mapping for kernels
         */
        AddCurrentDensity(MappingDesc const cellDescription) : cellDescription(cellDescription)
        {
        }

        /** Smooth current density and add it to the electric field with the given coefficient
         *
         * @tparam T_JBox type of device data box with current density values
         * @tparam T_CurrentInterpolationFunctor current interpolation functor type
         *
         * @param dataBoxJ device data box with current density values
         * @param currentInterpolationFunctor current interpolation functor
         * @param coeff coefficient to be used in the current interpolation functor
         */
        template<typename T_JBox, typename T_CurrentInterpolationFunctor>
        HINLINE void operator()(
            T_JBox dataBoxJ,
            T_CurrentInterpolationFunctor currentInterpolationFunctor,
            float_X coeff) const
        {
            DataConnector& dc = Environment<>::get().DataConnector();
            auto fieldE = dc.get<FieldE>(FieldE::getName());
            auto fieldB = dc.get<FieldB>(FieldB::getName());
            auto const mapper = makeAreaMapper<T_area>(cellDescription);

            PMACC_LOCKSTEP_KERNEL(KernelAddCurrentDensity{})
                .config(mapper.getGridDim(), SuperCellSize{})(
                    fieldE->getDeviceDataBox(),
                    fieldB->getDeviceDataBox(),
                    dataBoxJ,
                    currentInterpolationFunctor,
                    coeff,
                    mapper);
        }

    private:
        //! Mapping for kernels
        MappingDesc const cellDescription;
    };

} // namespace picongpu::fields::maxwellSolver
