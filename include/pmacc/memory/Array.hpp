/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace memory
    {
        /** static sized array
         *
         * mimic the most parts of the `std::array`
         */
        template<typename T_Type, size_t T_size>
        struct Array
        {
            using value_type = T_Type;
            using size_type = size_t;
            using reference = value_type&;
            using const_reference = value_type const&;
            using pointer = value_type*;
            using const_pointer = value_type const*;

            /** get number of elements */
            HDINLINE
            constexpr size_type size() const
            {
                return T_size;
            }

            /** get maximum number of elements */
            HDINLINE
            constexpr size_type max_size() const
            {
                return T_size;
            }

            /** get the direct access to the internal data
             *
             * @{
             */
            HDINLINE
            pointer data()
            {
                return reinterpret_cast<pointer>(m_data);
            }

            HDINLINE
            const_pointer data() const
            {
                return reinterpret_cast<const_pointer>(m_data);
            }

            /** @} */

            /** default constructor
             *
             * all members are uninitialized
             */
            Array() = default;

            /** constructor
             *
             * initialize each member with the given value
             *
             * @param value element assigned to each member
             */
            template<typename... T_Args>
            HDINLINE Array(T_Args&&... args)
            {
                for(size_type i = 0; i < size(); ++i)
                    reinterpret_cast<T_Type*>(m_data)[i] = std::move(T_Type{std::forward<T_Args>(args)...});
            }

            /** get N-th value
             *
             * @tparam T_Idx any type which can be implicit casted to an integral type
             * @param idx index within the array
             *
             * @{
             */
            template<typename T_Idx>
            HDINLINE const_reference operator[](T_Idx const idx) const
            {
                return reinterpret_cast<T_Type const*>(m_data)[idx];
            }

            template<typename T_Idx>
            HDINLINE reference operator[](T_Idx const idx)
            {
                return reinterpret_cast<T_Type*>(m_data)[idx];
            }

            /** @} */

        private:
            /** data storage
             *
             * std::array is a so-called "aggregate" which does not default-initialize
             * its members. In order to allow arbitrary types to skip implementing
             * a default constructor, this member is not stored as
             * `value_type m_data[ T_size ]` but as type-size aligned Byte type.
             */
            uint8_t m_data alignas(alignof(T_Type))[T_size * sizeof(T_Type)];
        };

    } // namespace memory
} // namespace pmacc
