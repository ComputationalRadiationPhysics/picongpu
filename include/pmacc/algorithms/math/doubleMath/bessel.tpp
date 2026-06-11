/*
 * SPDX-FileCopyrightText: Alexander Debus
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

#include <cmath>

namespace pmacc
{
    namespace math
    {
        namespace bessel
        {
            template<>
            struct I0<double>
            {
                using result = double;

                HDINLINE result operator()(result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::cyl_bessel_i0(x);
#else
                    return std::cyl_bessel_i(0, x);
#endif
                }
            };

            template<>
            struct I1<double>
            {
                using result = double;

                HDINLINE result operator()(result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::cyl_bessel_i1(x);
#else
                    return std::cyl_bessel_i(1, x);
#endif
                }
            };

            template<>
            struct J0<double>
            {
                using result = double;

                HDINLINE result operator()(result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::j0(x);
#else
                    return std::cyl_bessel_j(0, x);
#endif
                }
            };

            template<>
            struct J1<double>
            {
                using result = double;

                HDINLINE result operator()(result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::j1(x);
#else
                    return std::cyl_bessel_j(1, x);
#endif
                }
            };

            template<>
            struct Jn<int, double>
            {
                using result = double;

                HDINLINE result operator()(int const& n, result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::jn(n, x);
#else
                    return std::cyl_bessel_j(n, x);
#endif
                }
            };

            template<>
            struct Y0<double>
            {
                using result = double;

                HDINLINE result operator()(result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::y0(x);
#else
                    return std::cyl_neumann(0, x);
#endif
                }
            };

            template<>
            struct Y1<double>
            {
                using result = double;

                HDINLINE result operator()(result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::y1(x);
#else
                    return std::cyl_neumann(1, x);
#endif
                }
            };

            template<>
            struct Yn<int, double>
            {
                using result = double;

                HDINLINE result operator()(int const& n, result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::yn(n, x);
#else
                    return std::cyl_neumann(n, x);
#endif
                }
            };

        } // namespace bessel
    } // namespace math
} // namespace pmacc
