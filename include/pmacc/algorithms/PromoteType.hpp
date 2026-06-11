/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

namespace pmacc
{
    namespace algorithms
    {
        namespace promoteType
        {
            // general: use first type
            template<class T1, class T2>
            struct promoteType
            {
                using type = T1;
            };

            // special: promote float to double
            template<>
            struct promoteType<float, double>
            {
                using type = double;
            };


        } // namespace promoteType
    } // namespace algorithms
} // namespace pmacc
