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
 * Compare the analytic derivative of the Lehe dispersion relation with a central finite difference at several wave
 * numbers. The test covers all three Cherenkov-free axes with isotropic and anisotropic propagation directions.
 * The implemented relation follows Eq. (8), with coefficients from Eq. (11), of R. Lehe et al., "Numerical growth of
 * emittance in simulations of laser-wakefield acceleration", Phys. Rev. ST Accel. Beams 16, 021301 (2013),
 * https://doi.org/10.1103/PhysRevSTAB.16.021301. The angular frequency, wave numbers, and directions are chosen test
 * inputs, not tabulated paper data; reference derivatives are central differences of relation(), testing consistency
 * with the analytic derivative rather than independently validating the dispersion relation.
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
    //! Step for the central finite difference of relation()
    constexpr double h = 1e-6;

    //! Arbitrary angular frequency for the tested relation
    constexpr double omega = 2.0;

    //! Wave numbers to compare the analytic and finite-difference derivatives at
    std::vector<double> const kValues = {0.3, 0.7, 1.1, 1.7, 2.3, 2.9};

    template<uint32_t T_CherenkovFreeDir>
    void checkRelationDerivative(picongpu::float3_64 const& direction)
    {
        using Solver = picongpu::fields::maxwellSolver::Lehe<T_CherenkovFreeDir>;
        using Relation = picongpu::fields::maxwellSolver::DispersionRelation<Solver>;

        auto const relation = Relation{omega, direction};
        for(auto const k : kValues)
        {
            auto const finiteDifference = (relation.relation(k + h) - relation.relation(k - h)) / (2.0 * h);
            CHECK(relation.relationDerivative(k) == Catch::Approx(finiteDifference).epsilon(1e-6));
        }
    }
} // namespace

TEST_CASE("Lehe dispersion relation derivative", "[Lehe][DispersionRelation]")
{
    auto const isotropicDirection
        = picongpu::float3_64{1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
    auto const anisotropicDirection
        = picongpu::float3_64{1.0 / std::sqrt(14.0), 2.0 / std::sqrt(14.0), 3.0 / std::sqrt(14.0)};

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
    SECTION("Cherenkov-free direction 2")
    {
        checkRelationDerivative<2>(isotropicDirection);
        checkRelationDerivative<2>(anisotropicDirection);
    }
}
