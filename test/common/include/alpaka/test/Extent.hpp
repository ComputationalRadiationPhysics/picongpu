/**
 * \file
 * Copyright 2018 Benjamin Worpitz
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
