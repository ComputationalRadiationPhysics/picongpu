/* Copyright 2025-2026 Tapish Narwal, Luca Pennati, Rene Widera, Edgar Marquardt
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

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldTmpOperations.hpp"
#include "picongpu/fields/poissonSolver/FieldV.hpp"

#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/ExchangeMapping.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>

namespace picongpu::fields::poissonSolver
{
    struct BoundaryFunction
    {
        // gives the potential at the boundary, given the total cell coordinate for Dirichlet BCs and the outward
        // normal derivative for Neumann BCs change this function to implement different boundary conditions. for
        // Dirichlet BCs use units of potential, for Neumann BCs use units of potential per length. maybe make this a
        // param file parameter in the future
        HDINLINE auto operator()(
            math::Vector<double, 3u> const& totalCellCoordinate,
            DataSpace<3u> const& outwardNormal) const
        {
            return 0.0;
        }

        HDINLINE auto operator()(
            math::Vector<double, 2u> const& totalCellCoordinate,
            DataSpace<2u> const& outwardNormal) const
        {
            return 0.0;
        }
    };

    struct ApplyDirichletBCsFromFunctionKernel
    {
        DINLINE auto operator()(
            auto const& worker,
            auto fieldVBox,
            auto const boundaryFunction,
            DataSpace<simDim> cellOffsetToTotalOrigin,
            float_64 const& normalizationFactor,
            auto const mapper) const -> void
        {
            auto const relExchangeDir = Mask::getRelativeDirections<simDim>(mapper.getExchangeType());
            // including guards
            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));

            DataSpace<simDim> numGuardCells = mapper.getGuardingSuperCells() * SuperCellSize::toRT();

            // no guards included
            DataSpace<simDim> superCellTotalCellOffset
                = cellOffsetToTotalOrigin + superCellIdx * SuperCellSize::toRT() - numGuardCells;

            constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);

            forEachCellInSupercell(
                [&](int32_t const linearCellIdx)
                {
                    /* cell index within the superCell */
                    DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
                    // without guards
                    DataSpace<simDim> const totalCellIdx = superCellTotalCellOffset + cellIdx;

                    auto totalDistance = precisionCast<float_64>(totalCellIdx)
                                         * precisionCast<float_64>(sim.pic.getCellSize().shrink<simDim>());

                    fieldVBox(superCellIdx * SuperCellSize::toRT() + cellIdx)
                        = normalizationFactor * boundaryFunction(totalDistance, relExchangeDir);
                });
        }
    };

    struct ApplyDirichletBCsHomogeneousKernel
    {
        DINLINE auto operator()(
            auto const& worker,
            auto fieldVBox,
            auto const boundaryFunction,
            DataSpace<simDim> cellOffsetToTotalOrigin,
            float_64 const& normalizationFactor,
            auto const mapper) const -> void
        {
            // including guards
            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));

            constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);

            forEachCellInSupercell(
                [&](int32_t const linearCellIdx)
                {
                    /* cell index within the superCell */
                    DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);

                    fieldVBox(superCellIdx * SuperCellSize::toRT() + cellIdx) = 0.0;
                });
        }
    };

    struct ApplyNeumannBCsFromFunctionKernel
    {
        DINLINE auto operator()(
            auto const& worker,
            auto fieldVBox,
            auto const boundaryFunction,
            DataSpace<simDim> cellOffsetToTotalOrigin,
            float_64 const& normalizationFactor,
            auto const mapper) const -> void
        {
            auto const relExchangeDir = Mask::getRelativeDirections<simDim>(mapper.getExchangeType());
            // including guards
            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));

            DataSpace<simDim> const guardCells = mapper.getGuardingSuperCells() * SuperCellSize::toRT();

            // no guards included
            DataSpace<simDim> const localCells = mapper.getGridSuperCells() * SuperCellSize::toRT() - 2 * guardCells;
            DataSpace<simDim> const superCellCellOffset = superCellIdx * SuperCellSize::toRT();

            constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);

            forEachCellInSupercell(
                [&](int32_t const linearCellIdx)
                {
                    /* cell index within the superCell */
                    DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
                    // without guards
                    DataSpace<simDim> const totalCellIdx
                        = cellOffsetToTotalOrigin + superCellCellOffset - guardCells + cellIdx;

                    auto totalDistance = precisionCast<float_64>(totalCellIdx)
                                         * precisionCast<float_64>(sim.pic.getCellSize().shrink<simDim>());

                    DataSpace<simDim> const guardCellIdx = superCellCellOffset + cellIdx;
                    DataSpace<simDim> sourceCellIdx = guardCellIdx;

                    uint32_t dir = simDim;
                    for(uint32_t d = 0; d < simDim; ++d)
                    {
                        if(relExchangeDir[d] < 0)
                        {
                            sourceCellIdx[d] = guardCells[d];
                            dir = d;
                        }
                        else if(relExchangeDir[d] > 0)
                        {
                            sourceCellIdx[d] = guardCells[d] + localCells[d] - 1;
                            dir = d;
                        }
                    }

                    if(dir < simDim)
                    {
                        fieldVBox(guardCellIdx) = fieldVBox(sourceCellIdx)
                                                  + normalizationFactor
                                                        * boundaryFunction(totalDistance, relExchangeDir)
                                                        * sim.pic.getCellSize().shrink<simDim>()[dir];
                    }
                    else
                    {
                        fieldVBox(guardCellIdx) = fieldVBox(guardCellIdx);
                    }
                });
        }
    };

    struct ApplyNeumannBCsHomogeneousKernel
    {
        DINLINE auto operator()(
            auto const& worker,
            auto fieldVBox,
            auto const boundaryFunction,
            DataSpace<simDim> cellOffsetToTotalOrigin,
            float_64 const& normalizationFactor,
            auto const mapper) const -> void
        {
            auto const relExchangeDir = Mask::getRelativeDirections<simDim>(mapper.getExchangeType());
            // including guards
            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));

            DataSpace<simDim> const guardCells = mapper.getGuardingSuperCells() * SuperCellSize::toRT();

            // no guards included
            DataSpace<simDim> const localCells = mapper.getGridSuperCells() * SuperCellSize::toRT() - 2 * guardCells;
            DataSpace<simDim> const superCellCellOffset = superCellIdx * SuperCellSize::toRT();

            constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);

            forEachCellInSupercell(
                [&](int32_t const linearCellIdx)
                {
                    /* cell index within the superCell */
                    DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);


                    DataSpace<simDim> const guardCellIdx = superCellCellOffset + cellIdx;
                    DataSpace<simDim> sourceCellIdx = guardCellIdx;

                    for(uint32_t d = 0; d < simDim; ++d)
                    {
                        if(relExchangeDir[d] < 0)
                        {
                            sourceCellIdx[d] = guardCells[d];
                        }
                        else if(relExchangeDir[d] > 0)
                        {
                            sourceCellIdx[d] = guardCells[d] + localCells[d] - 1;
                        }
                    }
                    fieldVBox(guardCellIdx) = fieldVBox(sourceCellIdx);
                });
        }
    };

    template<typename T_Kernel, typename T_Buffer>
    void applyBoundaryConditions(
        T_Buffer& buffer,
        MappingDesc const& cellDescription,
        float_64 const& normalizationFactor = 1.0)
    {
        SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
        auto const globalDomain = subGrid.getGlobalDomain();
        auto const localDomain = subGrid.getLocalDomain();
        auto const cellOffsetToTotalOrigin = globalDomain.offset + localDomain.offset;

        for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
        {
            /* only call for planes: left, right, top, bottom, back, front */
            if(FRONT % i == 0 && !Environment<simDim>::get().GridController().getCommunicationMask().isSet(i))
            {
                ExchangeMapping<GUARD, MappingDesc> mapper(cellDescription, i);

                PMACC_LOCKSTEP_KERNEL(T_Kernel{})
                    .config(mapper.getGridDim(), SuperCellSize{})(
                        buffer.getDeviceBuffer().getDataBox(),
                        BoundaryFunction{},
                        cellOffsetToTotalOrigin,
                        normalizationFactor,
                        mapper);
            }
        }
    }

    namespace detail
    {
        inline std::tuple<float_64, float_64> computeEigenBounds(
            auto const& globalDomainSize,
            auto const& cellSize,
            bool const nonPeriodicNeumann)
        {
            float_64 dirichletMinSum = 0.0;
            float_64 nonDirichletMinPositive = 0.0;
            float_64 eigenMax = 0.0;
            bool hasDirichletAxis = false;

            auto const periodic
                = pmacc::Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();

            for(uint32_t d = 0; d < simDim; ++d)
            {
                // N must be the number of unique, owned scalar vertices
                // represented by this operator in dimension d.
                auto const nInt = globalDomainSize[d];
                if(nInt < 1)
                    throw std::runtime_error("Poisson eigenBounds requires at least one vertex per dimension");

                float_64 const n = static_cast<float_64>(nInt);
                float_64 const h = static_cast<float_64>(cellSize[d]);
                float_64 const invH2 = 1.0 / (h * h);
                float_64 minPositive = 0.0;
                float_64 maxEigenvalue = 0.0;

                if(periodic[d])
                {
                    if(nInt < 2)
                        throw std::runtime_error("Periodic Poisson direction requires at least two unique vertices");

                    // Periodic modes: theta_k = 2 pi k / N.
                    minPositive
                        = 4.0 * pmacc::math::pow(pmacc::math::sin(pmacc::math::Pi<float_64>::value / n), 2.0) * invH2;

                    // floor(N/2) is integer division here.
                    float_64 const highestMode = static_cast<float_64>(nInt / 2);
                    maxEigenvalue
                        = 4.0
                          * pmacc::math::pow(pmacc::math::sin(pmacc::math::Pi<float_64>::value * highestMode / n), 2.0)
                          * invH2;
                }
                else if(nonPeriodicNeumann)
                {
                    if(nInt < 2)
                        throw std::runtime_error("Neumann Poisson direction requires at least two vertices");

                    // Mirrored-ghost Neumann closure: theta_k = pi k / N.
                    minPositive = 4.0
                                  * pmacc::math::pow(pmacc::math::sin(pmacc::math::Pi<float_64>::halfValue / n), 2.0)
                                  * invH2;

                    maxEigenvalue = 4.0
                                    * pmacc::math::pow(
                                        pmacc::math::sin(pmacc::math::Pi<float_64>::halfValue * (n - 1.0) / n),
                                        2.0)
                                    * invH2;
                }
                else
                {
                    // Homogeneous Dirichlet ghost values: theta_k = pi k / (N + 1).
                    float_64 const denominator = n + 1.0;
                    minPositive
                        = 4.0
                          * pmacc::math::pow(pmacc::math::sin(pmacc::math::Pi<float_64>::halfValue / denominator), 2.0)
                          * invH2;

                    maxEigenvalue = 4.0
                                    * pmacc::math::pow(
                                        pmacc::math::sin(pmacc::math::Pi<float_64>::halfValue * n / denominator),
                                        2.0)
                                    * invH2;

                    dirichletMinSum += minPositive;
                    hasDirichletAxis = true;
                }

                eigenMax += maxEigenvalue;

                // For periodic/Neumann axes, the constant mode has eigenvalue zero.
                // Keep the smallest positive mode for the mean-zero subspace.
                if(!periodic[d] && !nonPeriodicNeumann)
                    continue;

                if(nonDirichletMinPositive == 0.0 || minPositive < nonDirichletMinPositive)
                    nonDirichletMinPositive = minPositive;
            }

            // Dirichlet axes remove the global constant nullspace. If all axes
            // are periodic/Neumann, this lower bound assumes the constant mode
            // is projected out of the solver vectors.
            float_64 const eigenMin = hasDirichletAxis ? dirichletMinSum : nonDirichletMinPositive;

            if(!(eigenMin > 0.0) || !(eigenMax >= eigenMin))
                throw std::runtime_error("Invalid Poisson preconditioner eigenvalue bounds");

            return std::make_tuple(eigenMin, eigenMax);
        }
    } // namespace detail

    struct BoundaryConditionsDirichlet
    {
        // Solver work buffers use homogeneous BCs; FieldV uses BoundaryFunction below.
        template<typename T_Buffer>
        void operator()(T_Buffer& buffer, MappingDesc const& cellDescription) const
        {
            applyBoundaryConditions<ApplyDirichletBCsHomogeneousKernel>(buffer, cellDescription);
        }

        void operator()(FieldV& fieldV, MappingDesc const& cellDescription, float_64 const& normalizationFactor = 1.0)
            const
        {
            applyBoundaryConditions<ApplyDirichletBCsFromFunctionKernel>(
                *fieldV.fieldVBuffer,
                cellDescription,
                normalizationFactor);
        }

        std::tuple<float_64, float_64> eigenBounds(auto const& globalDomainSize, auto const& cellSize) const
        {
            return detail::computeEigenBounds(globalDomainSize, cellSize, false);
        }

        bool needsSpecificMeans() const
        {
            auto const periodic
                = pmacc::Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();
            bool hasDirichletAxis = false;
            for(uint32_t d = 0; d < simDim; ++d)
            {
                if(!periodic[d])
                    hasDirichletAxis = true;
            }
            return !hasDirichletAxis; // Dirichlet BCs require rho to have mean zero only in fully periodic conditions.
        }

        float_64 specificRhoMean(auto const& globalDomainSize, auto const& cellSize) const
        {
            return 0.0; // Dirichlet BCs require rho to have mean zero only in fully periodic conditions.
        }
    };

    struct BoundaryConditionsHomogenousDirichlet : BoundaryConditionsDirichlet
    {
        void operator()(FieldV& fieldV, MappingDesc const& cellDescription, float_64 const& normalizationFactor = 1.0)
            const
        {
            applyBoundaryConditions<ApplyDirichletBCsHomogeneousKernel>(
                *fieldV.fieldVBuffer,
                cellDescription,
                normalizationFactor);
        }
    };

    struct BoundaryConditionsNeumann
    {
        // Solver work buffers use homogeneous BCs; FieldV uses BoundaryFunction below.
        template<typename T_Buffer>
        void operator()(T_Buffer& buffer, MappingDesc const& cellDescription) const
        {
            applyBoundaryConditions<ApplyNeumannBCsHomogeneousKernel>(buffer, cellDescription);
        }

        void operator()(FieldV& fieldV, MappingDesc const& cellDescription, float_64 const& normalizationFactor = 1.0)
            const
        {
            applyBoundaryConditions<ApplyNeumannBCsFromFunctionKernel>(
                *fieldV.fieldVBuffer,
                cellDescription,
                normalizationFactor);
        }

        std::tuple<float_64, float_64> eigenBounds(auto const& globalDomainSize, auto const& cellSize) const
        {
            return detail::computeEigenBounds(globalDomainSize, cellSize, true);
        }

        bool needsSpecificMeans() const
        {
            return true; // Neumann BCs require rho to have specific mean.
        }

        float_64 specificRhoMean(auto const& globalDomainSize, auto const& cellSize) const
        {
            auto const periodic
                = pmacc::Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();
            auto const spacing = cellSize.template shrink<simDim>();
            auto const numVertices = globalDomainSize.productOfComponents();

            BoundaryFunction const boundaryFunction{};
            float_64 boundaryFluxSum = 0.0;

            for(uint32_t d = 0; d < simDim; ++d)
            {
                // Periodic dimensions have no physical boundary faces.
                if(periodic[d])
                    continue;

                auto transverseSize = globalDomainSize;
                transverseSize[d] = 1;

                uint32_t const faceNodeCount = transverseSize.productOfComponents();

                for(uint32_t side = 0; side < 2; ++side)
                {
                    DataSpace<simDim> normal = DataSpace<simDim>::create(0);
                    normal[d] = side == 0 ? -1 : 1;

                    uint32_t const faceIndex = side == 0 ? 0 : globalDomainSize[d] - 1;

                    for(uint32_t linear = 0; linear < faceNodeCount; ++linear)
                    {
                        DataSpace<simDim> vertexIndex
                            = pmacc::math::mapToND(transverseSize, static_cast<int32_t>(linear));
                        vertexIndex[d] = faceIndex;

                        auto const position = precisionCast<float_64>(vertexIndex) * spacing;

                        // BoundaryFunction returns the outward-normal derivative d(phi)/dn.
                        float_64 const outwardDerivative = boundaryFunction(position, normal);

                        boundaryFluxSum += outwardDerivative / spacing[d];
                    }
                }
            }

            return -sim.pic.getEps0() * boundaryFluxSum / static_cast<float_64>(numVertices);
        }
    };

    struct BoundaryConditionsHomogenousNeumann : BoundaryConditionsNeumann
    {
        void operator()(FieldV& fieldV, MappingDesc const& cellDescription, float_64 const& normalizationFactor = 1.0)
            const
        {
            applyBoundaryConditions<ApplyNeumannBCsHomogeneousKernel>(
                *fieldV.fieldVBuffer,
                cellDescription,
                normalizationFactor);
        }

        float_64 specificRhoMean(auto const& globalDomainSize, auto const& cellSize) const
        {
            return 0.0; // Homogeneous Neumann BCs require rho to have mean zero.
        }
    };

    // Use Neumann boundary conditions for the Poisson solver by default.
    // Change this to BoundaryConditionsDirichlet to use Dirichlet boundary conditions instead.
    // maybe make this a param file parameter in the future
    using BoundaryConditions = BoundaryConditionsNeumann;
} // namespace picongpu::fields::poissonSolver
