/* Copyright 2016-2017 Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once


#include "pmacc_types.hpp"
#include "memory/Array.hpp"


namespace PMacc
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
    template<
        uint32_t T_uniqueId,
        typename T_Type
    >
    struct Allocate
    {
        /** get a shared memory
         *
         * @return reference to shared memory
         */
        static
        DINLINE T_Type &
        get()
        {
            __shared__ uint8_t smem alignas( alignof( T_Type ) ) [ sizeof( T_Type ) ];
            return *( reinterpret_cast< T_Type* >( smem ) );
        }
    };

    /** allocate shared memory
     *
     * shared memory is always uninitialized
     *
     * @tparam T_uniqueId unique id for this object
     *          (is needed if more than one instance of shared memory in one kernel is used)
     * @tparam T_Type type of the stored object
     * @return reference to shared memory
     *
     * @{
     */
    template<
        uint32_t T_uniqueId,
        typename T_Type
    >
    DINLINE T_Type&
    allocate( )
    {
        return Allocate<
            T_uniqueId,
            T_Type
        >::get( );
    }

    /* @param instance of the type to store (is not to initialize the shared memory) */
    template<
        uint32_t T_uniqueId,
        typename T_Type
    >
    DINLINE T_Type&
    allocate( T_Type const & )
    {
        return Allocate<
            T_uniqueId,
            T_Type
        >::get( );
    }
    /** @} */

} // namespace shared
} // namespace memory
} // namespace PMacc

/** allocate shared memory
 *
 * @warning Do not use this macro within a factory method. This macro contains
 * a pre compiler counter which is only increased once per source code line.
 * You must not use this macro to create an aliased pointer to shared memory.
 * Please use `PMacc::memory::shared::allocate< id, type >()` to create shared memory
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
 *         PMACC_SMEM( sharedMem, int );
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
 *         auto& sharedMem = PMacc::memory::shared::allocate< T_id, int >()
 *         return sharedMem;
 *     }
 * };
 * @endcode
 *
 * @param varName name of the variable
 * @param ... type of the variable
 */
#define PMACC_SMEM( varName, ... ) auto & varName = PMacc::memory::shared::allocate< __COUNTER__, __VA_ARGS__ >( )
