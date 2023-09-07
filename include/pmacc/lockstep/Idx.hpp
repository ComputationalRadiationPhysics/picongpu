/* Copyright 2017-2023 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"


namespace pmacc
{
    namespace lockstep
    {
        template<typename T_Type, typename T_Config>
        struct Variable;

        //! Hold current index within a lockstep domain
        struct Idx
        {
            /** Constructor
             *
             * @param domElemIndex linear index within the domain
             * @param workerElemIndex virtual workers linear index of the work item
             */
            HDINLINE Idx(uint32_t const domElemIndex, uint32_t const workerElemIndex)
                : workerElemIdx(std::move(workerElemIndex))
                , domElemIdx(std::move(domElemIndex))
            {
            }

            /** Get linear index
             *
             * @return range [0,domain size)
             */
            HDINLINE operator uint32_t() const
            {
                return domElemIdx;
            }

            template<typename T_Type, typename T_Config>
            friend struct Variable;

        private:
            /** N-th element the worker is processing */
            HDINLINE uint32_t getWorkerElemIdx() const
            {
                return workerElemIdx;
            }

            //! virtual workers linear index of the work item
            PMACC_ALIGN(workerElemIdx, uint32_t const);
            //! linear index within the domain
            PMACC_ALIGN(domElemIdx, uint32_t const);
        };


    } // namespace lockstep
} // namespace pmacc
