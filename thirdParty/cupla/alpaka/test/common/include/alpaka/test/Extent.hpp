/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The test specifics.
    namespace test
    {
        //#############################################################################
        //! 1D: (5)
        //! 2D: (5, 4)
        //! 3D: (5, 4, 3)
        //! 4D: (5, 4, 3, 2)
        // We have to be careful with the extents used.
        // When TIdx is a 8 bit signed integer and Dim is 4, the extent is extremely limited.
        template<
            std::size_t Tidx>
        struct CreateExtentBufVal
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TIdx>
            ALPAKA_FN_HOST_ACC
            static auto create(
                TIdx)
            -> TIdx
            {
                return static_cast<TIdx>(5u - Tidx);
            }
        };

        //#############################################################################
        //! 1D: (4)
        //! 2D: (4, 3)
        //! 3D: (4, 3, 2)
        //! 4D: (4, 3, 2, 1)
        template<
            std::size_t Tidx>
        struct CreateExtentViewVal
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TIdx>
            ALPAKA_FN_HOST_ACC
            static auto create(
                TIdx)
            -> TIdx
            {
                return static_cast<TIdx>(4u - Tidx);
            }
        };
    }
}
