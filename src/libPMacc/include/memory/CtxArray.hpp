/**
 * Copyright 2017 Rene Widera
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
#include "mappings/threads/ForEachIdx.hpp"
#include "mappings/threads/IdxConfig.hpp"

#include <type_traits>
#include <utility>

namespace PMacc
{
namespace memory
{
    /** static sized array
     *
     * The array is designed to hold context variables for the lock step programming
     * model. The number of elements depend on the index domain size and
     * the number of worker to process the indices.
     */
    template<
        typename T_Type,
        typename T_IdxConfig
    >
    struct CtxArray :
        public Array<
            T_Type,
            T_IdxConfig::numCollIter * T_IdxConfig::simdSize
        >,
        T_IdxConfig
    {

        using T_IdxConfig::domainSize;
        using T_IdxConfig::workerSize;
        using T_IdxConfig::simdSize;
        using T_IdxConfig::numCollIter;

        using BaseArray = Array<
            T_Type,
            T_IdxConfig::numCollIter * T_IdxConfig::simdSize
        >;

        /** default constructor
         *
         * data member are uninitialized
         */
        HDINLINE CtxArray() = default;

        /** constructor
         *
         * initialize each member with the given value
         *
         * @param value element assigned to each member
         */
        HDINLINE explicit CtxArray( T_Type const & value ) : BaseArray( value )
        {
        }

        /** disable copy constructor
         */
        HDINLINE CtxArray( CtxArray const & ) = delete;

        /** constructor
         *
         * initialize each member with the result of the given functor
         *
         * @tparam T_Functor type of the user functor
         * @tparam T_Args type of user parameters
         * @param workerIdx number of worker range: [0;workerSize)
         * @param functor functor to initialize the member ( need to implement `::operator(size_type idx)`)
         * @param args user defined arguments those should forwarded to the functor
         */
        template<
            typename T_Functor,
            typename ... T_Args
        >
        HDINLINE explicit CtxArray( uint32_t const workerIdx, T_Functor const & functor, T_Args const && ... args )
        {
            mappings::threads::ForEachIdx< T_IdxConfig >
            { workerIdx }(
                [&,this]( uint32_t const linearIdx, uint32_t const idx )
                {
                    (*this)[idx] = functor( linearIdx, idx, std::forward(args) ... );
                }
            );
        }

    };

} // namespace memory
} // namespace PMacc
