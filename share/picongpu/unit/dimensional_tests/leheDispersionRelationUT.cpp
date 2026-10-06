/* Copyright 2026 Rene Widera
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

/** @file
 * Check the Lehe dispersion relation and its derivative on unequal cell sizes in 2D and 3D, for every active
 * Cherenkov-free axis and oblique propagation. The independent reference reduces Eq. (8) using the coefficients
 * of Eqs. (10)-(11) in R. Lehe et al., "Numerical growth of emittance in simulations of laser-wakefield acceleration",
 * Phys. Rev. ST Accel. Beams 16, 021301 (2013), https://doi.org/10.1103/PhysRevSTAB.16.021301.
 * Wave numbers, frequency, and directions are chosen test inputs, not tabulated paper data. Reference derivatives
 * are central differences of this independent formula; a separate check retains consistency with relation().
 */

#include <pmacc/boost_workaround.hpp>

#include "picongpu/fields/MaxwellSolver/DispersionRelation.hpp"
#include "picongpu/fields/MaxwellSolver/Lehe/Lehe.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
    static_assert(picongpu::simDim == TEST_DIM);

    //! Step for the central finite difference of relation()
    constexpr double h = 1e-6;

    //! Arbitrary angular frequency for the tested relation
    constexpr double omega = 2.0;

    //! Wave numbers to compare the analytic and finite-difference derivatives at
    std::vector<double> const kValues = {0.3, 0.7, 1.1, 1.7, 2.3, 2.9};

    template<uint32_t T_CherenkovFreeDir>
    double referenceRelation(double const k, picongpu::float3_64 const& direction)
    {
        auto const step = picongpu::precisionCast<double>(picongpu::sim.pic.getCellSize());
        auto const dt = static_cast<double>(picongpu::fields::maxwellSolver::getTimeStep());
        auto const cdt = picongpu::sim.pic.getSpeedOfLight() * dt;
        auto const r = cdt / step[T_CherenkovFreeDir];
        auto const sine = std::sin(0.5 * std::acos(-1.0) * r);
        auto const delta = 0.25 * (1.0 - sine * sine / (r * r));
        auto q = picongpu::float3_64::create(0.0);
        auto transverse = 0.0;
        for(uint32_t d = 0; d < picongpu::simDim; ++d)
        {
            auto const s = std::sin(0.5 * k * direction[d] * step[d]);
            q[d] = s * s;
            if(d != T_CherenkovFreeDir)
                transverse += q[d] / (step[d] * step[d]);
        }
        // Eq. (11) reduces each mixed coefficient in Eq. (8) to 1 / step[d]^2.
        auto const q0 = q[T_CherenkovFreeDir];
        auto const h0 = step[T_CherenkovFreeDir];
        auto const temporal = std::sin(0.5 * omega * dt) / cdt;
        return (q0 - 4.0 * delta * q0 * q0) / (h0 * h0) + (1.0 - q0) * transverse - temporal * temporal;
    }

    template<uint32_t T_CherenkovFreeDir>
    void checkRelationDerivative(picongpu::float3_64 const& direction)
    {
        using Solver = picongpu::fields::maxwellSolver::Lehe<T_CherenkovFreeDir>;
        using Relation = picongpu::fields::maxwellSolver::DispersionRelation<Solver>;

        auto const relation = Relation{omega, direction};
        for(auto const k : kValues)
        {
            INFO("Cherenkov-free axis " << T_CherenkovFreeDir << ", k = " << k);
            auto const reference = referenceRelation<T_CherenkovFreeDir>(k, direction);
            CHECK(relation.relation(k) == Catch::Approx(reference).epsilon(1e-12).margin(1e-12));
            auto const referenceDerivative = (referenceRelation<T_CherenkovFreeDir>(k + h, direction)
                                              - referenceRelation<T_CherenkovFreeDir>(k - h, direction))
                                             / (2.0 * h);
            CHECK(relation.relationDerivative(k) == Catch::Approx(referenceDerivative).epsilon(1e-6).margin(1e-9));
            auto const finiteDifference = (relation.relation(k + h) - relation.relation(k - h)) / (2.0 * h);
            CHECK(relation.relationDerivative(k) == Catch::Approx(finiteDifference).epsilon(1e-6));
        }
    }
} // namespace

TEST_CASE("Lehe dispersion relation derivative", "[Lehe][DispersionRelation]")
{
    auto isotropicDirection = picongpu::float3_64::create(0.0);
    auto anisotropicDirection = picongpu::float3_64::create(0.0);
    double const normSquared = picongpu::simDim == 2u ? 5.0 : 14.0;
    for(uint32_t d = 0; d < picongpu::simDim; ++d)
    {
        isotropicDirection[d] = 1.0 / std::sqrt(static_cast<double>(picongpu::simDim));
        anisotropicDirection[d] = (d + 1.0) / std::sqrt(normSquared);
    }
    auto const step = picongpu::sim.pic.getCellSize();
    REQUIRE(step[0] != step[1]);
    REQUIRE(step[0] != step[2]);
    REQUIRE(step[1] != step[2]);

    SECTION("Cherenkov-free direction 0")
    {
        checkRelationDerivative<0>(isotropicDirection);
        checkRelationDerivative<0>(anisotropicDirection);
    }
    SECTION("Cherenkov-free direction 1")
    {
        checkRelationDerivative<1>(isotropicDirection);
        checkRelationDerivative<1>(anisotropicDirection);
    }
    if constexpr(picongpu::simDim == 3u)
    {
        SECTION("Cherenkov-free direction 2")
        {
            checkRelationDerivative<2>(isotropicDirection);
            checkRelationDerivative<2>(anisotropicDirection);
        }
    }
}
