/* Copyright 2017-2021 Heiko Burau
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
#include "pmacc/memory/Array.hpp"
#include <type_traits>
#include <limits>


namespace pmacc
{
    namespace memory
    {
        /** A memory pool of dynamic size containing indices.
         *
         * At initial state the pool consists of consecutive indices according to
         * the `size` parameter. A new index is created by calling `get()`.
         * If the user releases an index, by calling
         * `release()`, it will be recycled at the next `get()` call.
         * Therefore the initial ordering is not preserved.
         * This pool provides `begin()` and `end()` methods. The iteration is done
         * reversely, allowing for additions and removal of the current element while
         * iterating.
         *
         * Scalings:
         *  `<constructor>` ~ O(N)
         *  `get()`         ~ O(1)
         *  `release()`     ~ O(N)
         *  `<iterating>`   ~ O(N) ~ std::array
         *
         * @warning: This class is not thread-safe!
         *
         * @tparam T_Index type of index
         * @tparam T_maxSize maximum number of indices
         */
        template<typename T_Index, size_t T_maxSize>
        struct IndexPool
        {
        private:
            /** Reverse-iterator of the memory pool. The pool is iterated reversely
             * to ensure removal of the current element while iterating.
             */
            struct ReverseIterator
            {
                T_Index* pointer;

                HDINLINE
                ReverseIterator(T_Index* const pointer) : pointer(pointer)
                {
                }

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
                bool operator!=(ReverseIterator const& other) const
                {
                    return this->pointer != other.pointer;
                }
            };

            size_t m_size;
            Array<T_Index, T_maxSize> listIds;

        public:
            using Index = T_Index;

            PMACC_STATIC_ASSERT_MSG(std::numeric_limits<Index>::is_integer, _Index_type_must_be_an_integer_type);
            PMACC_STATIC_ASSERT_MSG(std::numeric_limits<Index>::is_signed, _Index_type_must_be_a_signed_type);
            PMACC_STATIC_ASSERT_MSG(T_maxSize > 0u, _maxSize_has_to_be_greater_than_zero);

            /** init pool with consecutive indices
             *
             * @param size initial number of indices
             */
            HDINLINE
            IndexPool(const Index size = 0) : m_size(size)
            {
                /* TODO: parallelize */
                for(size_t i = 0; i < T_maxSize; i++)
                    this->listIds[i] = static_cast<Index>(i);
            }

            /** get a new index */
            HDINLINE
            Index get()
            {
                if(this->m_size == T_maxSize - 1u)
                    return Index(-1);

                return this->listIds[this->m_size++];
            }

            /** release an index */
            HDINLINE
            void release(const Index idx)
            {
                /* find position of `idx` */
                size_t pos;
                for(size_t i = 0; i < this->m_size; i++)
                {
                    if(this->listIds[i] == idx)
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
            size_t size() const
            {
                return this->m_size;
            }

            /** get maximum number of indices within pool */
            HDINLINE
            constexpr size_t max_size() const
            {
                return T_maxSize;
            }

            HDINLINE
            ReverseIterator begin()
            {
                return ReverseIterator(this->listIds.data() + this->m_size - 1u);
            }

            HDINLINE
            ReverseIterator end()
            {
                return ReverseIterator(this->listIds.data() - 1u);
            }
        };

    } // namespace memory
} // namespace pmacc
