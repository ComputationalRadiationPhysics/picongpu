/* Copyright 2021-2023 Rene Widera, Sergei Bastrakov
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

#include <cstdint>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            namespace fdtd
            {
                /** Base stencil functor to update fields inside the kernel
                 *
                 * This class serves to define the interface requirements for stencil functor implementations.
                 * So if roughly defines a "concept".
                 *
                 * @tparam T_Curl curl functor type to be applied to update the destination field,
                 *                 adheres to the Curl concept
                 */
                template<typename T_Curl>
                struct StencilFunctor
                {
                public:
                    //! Stencil requirements for lower margins
                    using LowerMargin = typename traits::GetLowerMargin<T_Curl>::type;
                    //! Stencil requirements for upper margins
                    using UpperMargin = typename traits::GetUpperMargin<T_Curl>::type;

                    /** Update field at the given position
                     *
                     * @tparam T_SrcBox source box type
                     * @tparam T_DestBox destination box type
                     *
                     * @param gridIndex index of the updated field element, with guards
                     * @param srcBoc source box shifted to position gridIndex,
                     *               note that it is the box, not the value
                     * @param destBox destination box shifted to position gridIndex,
                     *               note that it is the box, not the value
                     *
                     * @return update the value pointed to by localE
                     */
                    template<typename T_SrcBox, typename T_DestBox>
                    DINLINE void operator()(
                        pmacc::DataSpace<simDim> const& gridIndex,
                        T_SrcBox const srcBoc,
                        T_DestBox destBox);
                };

            } // namespace fdtd
        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu
