/* Copyright 2017 Heiko Burau
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
#include <type_traits>

namespace PMacc
{
namespace memory
{

    /* A memory pool of dynamic size containing indices
     *
     * \tparam T_Index type of index
     * \tparam T_maxSize maximum number of indices
     *
     * Warning: This class is not thread-safe!
     */
    template<
        typename T_Index,
        size_t T_maxSize
    >
    struct IndexPool
    {
    private:

        /* Reverse-iterator of the memory pool. The pool is iterated reversely
         * to ensure removal of the current element while iterating.
         */
        struct ReverseIterator
        {
            T_Index* pointer;

            HDINLINE
            ReverseIterator( T_Index* const pointer ) : pointer( pointer )
            {}

            HDINLINE
            void operator++()
            {
                this->pointer--;
            }

            HDINLINE
            T_Index& operator*()
            {
                return *(this->pointer);
            }

            HDINLINE
            bool operator!=( const ReverseIterator& other ) const
            {
                return this->pointer != other.pointer;
            }
        };

        size_t m_size;
        Array<
            T_Index,
            T_maxSize
        > listIds;

    public:

        using Index = T_Index;

        /* init pool with consecutive indices
         *
         * \param size initial number of indices
         */
        HDINLINE
        IndexPool( const Index size = 0 ) : m_size( size )
        {
            /* TODO: parallelize */
            for( size_t i = 0; i < T_maxSize; i++ )
                this->listIds[i] = static_cast< Index >( i );
        }

        /* get a new index */
        HDINLINE
        Index get()
        {
            if( this->m_size == T_maxSize - 1 )
                return Index(-1);

            return this->listIds[this->m_size++];
        }

        /* release an index */
        HDINLINE
        void release( const Index idx )
        {
            /* find position of `idx` */
            size_t pos;
            for( size_t i = 0; i < this->m_size; i++ )
            {
                if( this->listIds[i] == idx )
                {
                    pos = i;
                    break;
                }
            }

            this->listIds[pos] = this->listIds[--this->m_size];
            this->listIds[this->m_size] = idx;
        }

        /** get number of indices within pool */
        HDINLINE
        size_t size( ) const
        {
            return this->m_size;
        }

        /** get maximum number of indices within pool */
        HDINLINE
        constexpr size_t max_size( ) const
        {
            return T_maxSize;
        }

        HDINLINE
        ReverseIterator begin()
        {
            return ReverseIterator( this->listIds.data() + this->m_size - 1 );
        }

        HDINLINE
        ReverseIterator end()
        {
            return ReverseIterator( this->listIds.data() - 1 );
        }
    };

} // namespace memory
} // namespace PMacc
