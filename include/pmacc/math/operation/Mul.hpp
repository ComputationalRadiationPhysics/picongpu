/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/mpi/GetMPI_Op.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace math
    {
        namespace operation
        {
            struct Mul
            {
                template<typename Dst, typename Src>
                HDINLINE constexpr void operator()(Dst& dst, Src const& src) const
                {
                    dst *= src;
                }

                template<typename Dst, typename Src, typename T_Worker>
                HDINLINE constexpr void operator()(T_Worker const&, Dst& dst, Src const& src) const
                {
                    dst *= src;
                }
            };
        } // namespace operation
    } // namespace math
} // namespace pmacc

namespace pmacc
{
    namespace mpi
    {
        template<>
        HINLINE MPI_Op getMPI_Op<pmacc::math::operation::Mul>()
        {
            return MPI_PROD;
        }
    } // namespace mpi
} // namespace pmacc
