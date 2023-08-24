/* Copyright 2016-2022 Rene Widera
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


#include "pmacc/lockstep/lockstep.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/types.hpp"


namespace pmacc
{
    namespace memory
    {
        namespace shared
        {
            /** allocate shared memory
             *
             * shared memory is always uninitialized
             *
             * @tparam T_uniqueId unique id for this object
             *          (is needed if more than one instance of shared memory in one kernel is used)
             * @tparam T_Type type of the stored object
             */
            template<uint32_t T_uniqueId, typename T_Type>
            struct Allocate
            {
                /** get a shared memory
                 *
                 * @return reference to shared memory
                 */
                template<typename T_Acc>
                DINLINE static T_Type& get(T_Acc const& acc)
                {
                    auto& smem = ::alpaka::declareSharedVar<T_Type, T_uniqueId>(acc);
                    return smem;
                }

                template<typename T_Acc, uint32_t T_numSuggestedWorkers>
                DINLINE static T_Type& get(pmacc::lockstep::Worker<T_Acc, T_numSuggestedWorkers> const& worker)
                {
                    auto& smem = ::alpaka::declareSharedVar<T_Type, T_uniqueId>(worker.getAcc());
                    return smem;
                }
            };

            /** allocate shared memory
             *
             * shared memory is always uninitialized
             *
             * @tparam T_uniqueId unique id for this object
             *          (is needed if more than one instance of shared memory in one kernel is used)
             * @tparam T_Type type of the stored object
             * @param accOrWorker alpaka accelerator or lockstep worker
             * @return reference to shared memory
             */
            template<uint32_t T_uniqueId, typename T_Type, typename T_AccOrWorker>
            DINLINE T_Type& allocate(T_AccOrWorker const& accOrWorker)
            {
                return Allocate<T_uniqueId, T_Type>::get(accOrWorker);
            }


        } // namespace shared
    } // namespace memory
} // namespace pmacc

/** allocate shared memory
 *
 * @warning Do not use this macro within a factory method. This macro contains
 * a pre compiler counter which is only increased once per source code line.
 * You must not use this macro to create an aliased pointer to shared memory.
 * Please use `pmacc::memory::shared::allocate< id, type >()` to create shared memory
 * which is used outside of the method where it is created.
 *
 * @code
 * // THIS IS A EXAMPLE HOW `PMACC_SMEM` SHOULD NOT BE USED
 * struct Factory
 * {
 *     int& getSharedMem( )
 *     {
 *         // this macro points always to the same memory address
 *         // even if this method is called twice
 *         PMACC_SMEM(accOrLockstepWorker, sharedMem, int);
 *         return sharedMem;
 *     }
 * };
 *
 * // RIGHT USAGE
 * template< uint32_t T_id >
 * struct Factory
 * {
 *     int& getSharedMem( )
 *     {
 *         // create new shared memory for each `T_id`
 *         auto& sharedMem = pmacc::memory::shared::allocate<T_id, int>(accOrLockstepWorker)
 *         return sharedMem;
 *     }
 * };
 * @endcode
 *
 * @param accOrWorker alpaka accelerator or lockstep worker
 * @param varName name of the variable
 * @param ... type of the variable
 */
#define PMACC_SMEM(accOrWorker, varName, ...)                                                                         \
    auto& varName = pmacc::memory::shared::allocate<__COUNTER__, __VA_ARGS__>(accOrWorker)
