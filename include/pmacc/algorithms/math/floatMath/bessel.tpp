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
            struct I0<float>
            {
                using result = float;

                HDINLINE result operator()(result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::cyl_bessel_i0f(x);
#else
                    return std::cyl_bessel_i(0.f, x);
#endif
                }
            };

            template<>
            struct I1<float>
            {
                using result = float;

                HDINLINE result operator()(result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::cyl_bessel_i1f(x);
#else
                    return std::cyl_bessel_i(1.f, x);
#endif
                }
            };

            template<>
            struct J0<float>
            {
                using result = float;

                HDINLINE result operator()(result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu_
                    return ::j0f(x);
#else
                    return std::cyl_bessel_j(0.f, x);
#endif
                }
            };

            template<>
            struct J1<float>
            {
                using result = float;

                HDINLINE result operator()(result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::j1f(x);
#else
                    return std::cyl_bessel_j(1.f, x);
#endif
                }
            };

            template<>
            struct Jn<int, float>
            {
                using result = float;

                HDINLINE result operator()(int const& n, result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::jnf(n, x);
#else
                    return std::cyl_bessel_j(static_cast<float>(n), x);
#endif
                }
            };

            template<>
            struct Y0<float>
            {
                using result = float;

                HDINLINE result operator()(result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::y0f(x);
#else
                    return std::cyl_neumann(0.f, x);
#endif
                }
            };

            template<>
            struct Y1<float>
            {
                using result = float;

                HDINLINE result operator()(result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::y1f(x);
#else
                    return std::cyl_neumann(1.f, x);
#endif
                }
            };

            template<>
            struct Yn<int, float>
            {
                using result = float;

                HDINLINE result operator()(int const& n, result const& x)
                {
#if (PMACC_DEVICE_COMPILE == 1) // we are on gpu
                    return ::ynf(n, x);
#else
                    return std::cyl_neumann(static_cast<float>(n), x);
#endif
                }
            };

        } // namespace bessel
    } // namespace math
} // namespace pmacc
