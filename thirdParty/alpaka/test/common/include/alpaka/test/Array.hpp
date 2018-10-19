/**
 * \file
 * Copyright 2017 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

namespace alpaka
{
    namespace test
    {
        //#############################################################################
        template<
            typename TType,
            size_t TSize>
        struct Array {
            TType m_data[TSize];

            template<
                typename T_Idx>
            ALPAKA_FN_HOST_ACC const TType &operator[](
                const T_Idx idx) const
            {
                return m_data[idx];
            }

            template<
                typename TIdx>
            ALPAKA_FN_HOST_ACC TType & operator[](
                const TIdx idx)
            {
                return m_data[idx];
            }
        };
    }
}
