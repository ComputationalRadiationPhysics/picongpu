/* Copyright 2026-2026 PIConGPU contributors
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "picongpu/fields/poissonSolver/Stencil.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{
    using picongpu::fields::poissonSolver::GetFieldEStencil;
    using picongpu::fields::poissonSolver::StencilFunc;
    using picongpu::float3_X;
    using picongpu::float_64;
    using picongpu::float_X;
    using pmacc::DataSpace;

    enum class Ghost
    {
        Dirichlet,
        Periodic,
        NeumannMirror
    };

    std::size_t index(int i, int j, int k, int nx, int ny)
    {
        return static_cast<std::size_t>(i + nx * (j + ny * k));
    }

    /** Minimal field accessor matching the interface expected by StencilFunc.
     *
     * The production stencil functors are HDINLINE (host/device), so this mock
     * must expose the same qualification. Without it nvcc warns that a
     * __host__ __device__ functor calls host-only accessor methods. Only the
     * host path is ever executed by the unit test; the device variants are
     * well-formed because the accessor stores plain scalars.
     */
    struct Accessor
    {
        float_64 center = 0.0;
        float_64 minus[3] = {0.0, 0.0, 0.0};
        float_64 plus[3] = {0.0, 0.0, 0.0};

        HDINLINE float_64 operator*() const
        {
            return center;
        }

        HDINLINE float_64 operator()(DataSpace<3u> const& offset) const
        {
            for(int d = 0; d < 3; ++d)
            {
                if(offset[d] > 0)
                    return plus[d];
                if(offset[d] < 0)
                    return minus[d];
            }
            return center;
        }
    };

    float_64 neighbour(
        std::vector<float_64> const& field,
        int nx,
        int ny,
        int nz,
        int i,
        int j,
        int k,
        int direction,
        int delta,
        Ghost ghost)
    {
        int idx[3] = {i, j, k};
        int const size[3] = {nx, ny, nz};
        idx[direction] += delta;
        if(idx[direction] < 0 || idx[direction] >= size[direction])
        {
            if(ghost == Ghost::Periodic)
                idx[direction] = (idx[direction] + size[direction]) % size[direction];
            else if(ghost == Ghost::NeumannMirror)
                idx[direction] = idx[direction] < 0 ? 0 : size[direction] - 1;
            else
                return 0.0; // homogeneous Dirichlet ghost
        }
        return field[index(idx[0], idx[1], idx[2], nx, ny)];
    }

    Accessor makeAccessor(std::vector<float_64> const& field, int nx, int ny, int nz, int i, int j, int k, Ghost ghost)
    {
        Accessor accessor;
        accessor.center = field[index(i, j, k, nx, ny)];
        for(int d = 0; d < 3; ++d)
        {
            accessor.minus[d] = neighbour(field, nx, ny, nz, i, j, k, d, -1, ghost);
            accessor.plus[d] = neighbour(field, nx, ny, nz, i, j, k, d, +1, ghost);
        }
        return accessor;
    }

    std::vector<float_64> applyStencil(
        std::vector<float_64> const& field,
        int nx,
        int ny,
        int nz,
        float_64 h,
        Ghost ghost)
    {
        std::vector<float_64> result(field.size());
        pmacc::math::Vector<float_X, 3u> const cellSizeSquared{
            static_cast<float_X>(h * h),
            static_cast<float_X>(h * h),
            static_cast<float_X>(h * h)};
        float_64 const fc0 = 2.0 * (1.0 / cellSizeSquared[0] + 1.0 / cellSizeSquared[1] + 1.0 / cellSizeSquared[2]);

        StencilFunc const stencil;
        for(int k = 0; k < nz; ++k)
            for(int j = 0; j < ny; ++j)
                for(int i = 0; i < nx; ++i)
                    result[index(i, j, k, nx, ny)] = stencil.stencilFunction(
                        makeAccessor(field, nx, ny, nz, i, j, k, ghost),
                        fc0,
                        cellSizeSquared);
        return result;
    }

    float_64 inner(std::vector<float_64> const& lhs, std::vector<float_64> const& rhs)
    {
        float_64 sum = 0.0;
        for(std::size_t i = 0; i < lhs.size(); ++i)
            sum += lhs[i] * rhs[i];
        return sum;
    }
} // namespace

TEST_CASE("Poisson stencil is exact for the Dirichlet polynomial", "[poisson][stencil]")
{
    constexpr int n = 5;
    float_64 const h = 0.5; // exactly representable in float_X
    float_64 const L = (static_cast<float_64>(n) + 1.0) * h;
    float_64 const phi0 = 1.0;

    std::vector<float_64> phi(n * n * n);
    for(int k = 0; k < n; ++k)
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < n; ++i)
            {
                float_64 const x = (static_cast<float_64>(i) + 1.0) * h;
                float_64 const y = (static_cast<float_64>(j) + 1.0) * h;
                float_64 const z = (static_cast<float_64>(k) + 1.0) * h;
                phi[index(i, j, k, n, n)]
                    = 64.0 * phi0 * x * (L - x) * y * (L - y) * z * (L - z) / (L * L * L * L * L * L);
            }

    auto const stencilPhi = applyStencil(phi, n, n, n, h, Ghost::Dirichlet);

    float_64 maxError = 0.0;
    float_64 maxExpected = 0.0;
    for(int k = 0; k < n; ++k)
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < n; ++i)
            {
                float_64 const x = (static_cast<float_64>(i) + 1.0) * h;
                float_64 const y = (static_cast<float_64>(j) + 1.0) * h;
                float_64 const z = (static_cast<float_64>(k) + 1.0) * h;
                float_64 const px = x * (L - x);
                float_64 const py = y * (L - y);
                float_64 const pz = z * (L - z);
                float_64 const expected = 128.0 * phi0 * (py * pz + px * pz + px * py) / (L * L * L * L * L * L);
                maxError = std::max(maxError, std::abs(stencilPhi[index(i, j, k, n, n)] - expected));
                maxExpected = std::max(maxExpected, std::abs(expected));
            }

    CHECK(maxError <= 1e-10 * maxExpected);
}

TEST_CASE("Poisson stencil is self-adjoint", "[poisson][stencil]")
{
    constexpr int nx = 4;
    constexpr int ny = 5;
    constexpr int nz = 3;
    float_64 const h = 0.7;

    std::vector<float_64> u(nx * ny * nz);
    std::vector<float_64> v(nx * ny * nz);
    for(int k = 0; k < nz; ++k)
        for(int j = 0; j < ny; ++j)
            for(int i = 0; i < nx; ++i)
            {
                std::size_t const idx = index(i, j, k, nx, ny);
                u[idx] = std::sin(0.7 * i + 0.3 * j + 0.2 * k) + 0.1 * i - 0.05 * k;
                v[idx] = std::cos(0.2 * i - 0.4 * j + 0.5 * k) + 0.03 * j;
            }

    for(Ghost ghost : {Ghost::Dirichlet, Ghost::Periodic, Ghost::NeumannMirror})
    {
        auto const Lu = applyStencil(u, nx, ny, nz, h, ghost);
        auto const Lv = applyStencil(v, nx, ny, nz, h, ghost);
        float_64 const lhs = inner(Lu, v);
        float_64 const rhs = inner(u, Lv);
        float_64 const scale = std::max({std::abs(lhs), std::abs(rhs), 1.0});
        CHECK(std::abs(lhs - rhs) <= 1e-11 * scale);
    }
}

TEST_CASE("Neumann mirror ghosts have a constant null space", "[poisson][stencil]")
{
    constexpr int n = 6;
    float_64 const h = 0.25;
    std::vector<float_64> constant(n * n * n, 2.5);

    auto const result = applyStencil(constant, n, n, n, h, Ghost::NeumannMirror);
    for(auto value : result)
        CHECK(std::abs(value) <= 1e-12);
}

TEST_CASE("Gradient and stencil satisfy the discrete adjoint identity", "[poisson][stencil]")
{
    constexpr int n = 5;
    float_64 const h = 0.3;

    std::vector<float_64> u(n * n * n);
    std::vector<float_64> w(n * n * n);
    for(int k = 0; k < n; ++k)
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < n; ++i)
            {
                std::size_t const idx = index(i, j, k, n, n);
                u[idx] = std::sin(0.4 * i + 0.2 * j + 0.1 * k);
                w[idx] = std::cos(0.3 * i - 0.5 * j + 0.2 * k);
            }

    pmacc::math::Vector<float_X, 3u> const cellSize{
        static_cast<float_X>(h),
        static_cast<float_X>(h),
        static_cast<float_X>(h)};
    GetFieldEStencil const gradient;

    // G u on every node with homogeneous Dirichlet ghosts.
    std::vector<float3_X> gradU(u.size());
    for(int k = 0; k < n; ++k)
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < n; ++i)
            {
                Accessor accessor;
                accessor.center = u[index(i, j, k, n, n)];
                for(int d = 0; d < 3; ++d)
                    accessor.plus[d] = neighbour(u, n, n, n, i, j, k, d, +1, Ghost::Dirichlet);
                gradU[index(i, j, k, n, n)] = gradient.getFieldE(accessor, cellSize);
            }

    // <G u, w> and <u, D w>, with D the backward difference.
    float_64 lhs = 0.0;
    float_64 rhs = 0.0;
    for(int k = 0; k < n; ++k)
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < n; ++i)
            {
                std::size_t const idx = index(i, j, k, n, n);
                float3_X const g = gradU[idx];
                lhs += g.x() * w[idx] + g.y() * w[idx] + g.z() * w[idx];

                float_64 const wx = w[idx] - neighbour(w, n, n, n, i, j, k, 0, -1, Ghost::Dirichlet);
                float_64 const wy = w[idx] - neighbour(w, n, n, n, i, j, k, 1, -1, Ghost::Dirichlet);
                float_64 const wz = w[idx] - neighbour(w, n, n, n, i, j, k, 2, -1, Ghost::Dirichlet);
                rhs += u[idx] * (wx + wy + wz) / h;
            }

    float_64 const scale = std::max({std::abs(lhs), std::abs(rhs), 1.0});
    CHECK(std::abs(lhs - rhs) <= 1e-5 * scale);

    // D G u must reproduce the 7-point stencil.
    auto const stencilU = applyStencil(u, n, n, n, h, Ghost::Dirichlet);
    for(int k = 0; k < n; ++k)
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < n; ++i)
            {
                std::size_t const idx = index(i, j, k, n, n);
                float_64 divergence = 0.0;
                for(int d = 0; d < 3; ++d)
                {
                    int prev[3] = {i, j, k};
                    prev[d] -= 1;

                    float_64 const currentComponent
                        = d == 0 ? gradU[idx].x() : (d == 1 ? gradU[idx].y() : gradU[idx].z());

                    // At a boundary node the previous node is the Dirichlet
                    // ghost, where G u = (u_ghost - u_0)/h = -u_0/h.
                    float_64 previousComponent = 0.0;
                    if(prev[d] < 0)
                        previousComponent = -u[idx] / h;
                    else
                    {
                        float3_X const previous = gradU[index(prev[0], prev[1], prev[2], n, n)];
                        previousComponent = d == 0 ? previous.x() : (d == 1 ? previous.y() : previous.z());
                    }
                    divergence += (currentComponent - previousComponent) / h;
                }
                CHECK(std::abs(divergence - stencilU[idx]) <= 1e-5 * std::max(std::abs(stencilU[idx]), 1.0));
            }
}
