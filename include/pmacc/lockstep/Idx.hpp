/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
