/* Copyright 2020 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cmath>

//! Exact solution to the test problem
//! u_t(x, y, t) = u_xx(x, t) + u_yy(y, t), x in [0, 1], y in [0, 1], t in [0, T]
//!
//! \param x value of x
//! \param x value of y
//! \param t value of t
ALPAKA_FN_HOST_ACC auto exactSolution(double const x, double const y, double const t) -> double
{
    constexpr double pi = alpaka::math::constants::pi;
    return std::exp(-pi * pi * t) * (std::sin(pi * x) + std::sin(pi * y));
}

//! Valdidate calculated solution in the buffer to the analytical solution at t=tMax
//!
//! \param buffer buffer holding the solution at t=tMax
//! \param extent extents of the buffer
//! \param dx
//! \param dy
//! \param tMax time at simulation end
template<typename T_Buffer, typename T_Extent>
auto validateSolution(
    T_Buffer const& buffer,
    T_Extent const& extent,
    double const dx,
    double const dy,
    double const tMax) -> std::pair<bool, double>
{
    // Calculate error
    double maxError = 0.0;
    for(uint32_t j = 1; j < extent[0] - 1; ++j)
    {
        for(uint32_t i = 1; i < extent[1] - 1; ++i)
        {
            auto const error = std::abs(buffer.data()[j * extent[1] + i] - exactSolution(i * dx, j * dy, tMax));
            maxError = std::max(maxError, error);
        }
    }

    constexpr double errorThreshold = 1e-4;
    return std::make_pair(maxError < errorThreshold, maxError);
}

//! Initialize the buffer to the analytical solution at t=0
//!
//! \param buffer buffer holding the solution at tMax
//! \param extent extents of the buffer
//! \param dx
//! \param dy
template<typename TBuffer>
auto initalizeBuffer(TBuffer& buffer, double const dx, double const dy) -> void
{
    auto extents = alpaka::getExtents(buffer);
    // Apply initial conditions for the test problem
    for(uint32_t j = 0; j < extents[0]; ++j)
    {
        for(uint32_t i = 0; i < extents[1]; ++i)
        {
            buffer.data()[j * extents[1] + i] = exactSolution(i * dx, j * dy, 0.0);
        }
    }
}
