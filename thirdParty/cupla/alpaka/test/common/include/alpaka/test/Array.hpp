/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace alpaka
{
    namespace test
    {
        //#############################################################################
        template<typename TType, size_t TSize>
        struct Array
        {
            TType m_data[TSize];

            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC const TType& operator[](const T_Idx idx) const
            {
                return m_data[idx];
            }

            template<typename TIdx>
            ALPAKA_FN_HOST_ACC TType& operator[](const TIdx idx)
            {
                return m_data[idx];
            }
        };
    } // namespace test
} // namespace alpaka
