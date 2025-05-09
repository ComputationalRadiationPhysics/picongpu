/* Copyright 2025 Tapish Narwal, Luca Pennati, Rene Widera
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

#include "picongpu/simulation/stage/Poisson.hpp"

#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/FieldJ.kernel"
#include "picongpu/fields/FieldTmpOperations.hpp"
#include "picongpu/fields/currentDeposition/Deposit.hpp"
#include "picongpu/fields/poissonSolver/BoundaryConditions.hpp"
#include "picongpu/fields/poissonSolver/RightHandSideNormalization.hpp"
#include "picongpu/fields/poissonSolver/Stencil.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/particleToGrid/CombinedDerive.hpp"
#include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.hpp"
#include "picongpu/simulation/stage/Poisson.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#include <pmacc/memory/boxes/DataBoxUnaryTransform.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

#include <picongpu/param/particle.param>

namespace picongpu::simulation::stage
{
    template<typename T_Func>
    struct DeviceLambda
    {
        T_Func const func;

        template<typename... T>
        DEVICEONLY auto operator()(T&&... args) const
        {
            return func(std::forward<T>(args)...);
        }

        template<typename... T>
        DEVICEONLY auto operator()(T&&... args)
        {
            return func(std::forward<T>(args)...);
        }
    };

    template<typename T_Func>
    DeviceLambda(T_Func const) -> DeviceLambda<T_Func>;
} // namespace picongpu::simulation::stage

namespace alpaka
{
    template<typename T>
    struct IsKernelArgumentTriviallyCopyable<picongpu::simulation::stage::DeviceLambda<T>, void> : std::true_type
    {
    };
} // namespace alpaka

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            namespace detail
            {

                template<typename T_SpeciesType, typename T_Area>
                struct ComputeChargeDensity
                {
                    using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
                    static uint32_t const area = T_Area::value;

                    HINLINE void operator()(FieldTmp& fieldTmp, uint32_t const currentStep) const
                    {
                        DataConnector& dc = Environment<>::get().DataConnector();

                        /* load species without copying the particle data to the host */
                        auto speciesTmp = dc.get<SpeciesType>(SpeciesType::FrameType::getName());

                        /* run algorithm */
                        using ChargeDensitySolver = typename particles::particleToGrid::CreateFieldTmpOperation_t<
                            SpeciesType,
                            particles::particleToGrid::derivedAttributes::ChargeDensity>::Solver;

                        computeFieldTmpValue<area, ChargeDensitySolver>(fieldTmp, *speciesTmp, currentStep);
                    }
                };

            } // namespace detail

            namespace deriveField = particles::particleToGrid;
            template<typename T>
            using SpeciesEligibleForChargeConservation = typename particles::traits::
                SpeciesEligibleForSolver<T, deriveField::derivedAttributes::ChargeDensity>::type;

            Poisson::Poisson(MappingDesc const mappingDesc)
                : m_mappingDesc(mappingDesc)
                , localReduce{std::make_unique<pmacc::device::Reduce>(1024)}
            {
                pkBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc.getGridLayout());
                rkBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc.getGridLayout());
                r0Buffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc.getGridLayout());
                mpkBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc.getGridLayout());
                ampkBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc.getGridLayout());
                zkBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc.getGridLayout());
                azkBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc.getGridLayout());
                fieldV = std::make_shared<fields::poissonSolver::FieldV>(m_mappingDesc);

                auto const commTag0 = pmacc::traits::getUniqueId<uint32_t>();
                auto const commTag1 = pmacc::traits::getUniqueId<uint32_t>();
                auto const commTagFieldV = pmacc::traits::getUniqueId<uint32_t>();
                /*go over all directions*/
                for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
                {
                    if(FRONT % i == 0)
                    {
                        DataSpace<simDim> relativeMask = Mask::getRelativeDirections<simDim>(i);
                        /* guarding cells depend on direction
                         * for negative direction use originGuard else endGuard (relative direction ZERO is ignored)
                         * don't switch end and origin because this is a read buffer and no send buffer
                         */
                        auto guardingCells = DataSpace<simDim>::create(0);
                        for(uint32_t d = 0; d < simDim; ++d)
                            guardingCells[d] = (relativeMask[d] == 0 ? 0 : 1);
                        mpkBuffer->addExchange(GUARD, i, guardingCells, commTag0);
                        zkBuffer->addExchange(GUARD, i, guardingCells, commTag1);
                        fieldV->fieldVBuffer->addExchange(GUARD, i, guardingCells, commTagFieldV);
                    }
                }
                DataConnector& dc = Environment<>::get().DataConnector();
                dc.share(fieldV);

                participate(true);
            }

            template<typename T_TranformFunctor>
            class TransformDataBox : private T_TranformFunctor
            {
            public:
                using ValueType = decltype(std::declval<T_TranformFunctor>()(DataSpace<simDim>::create(0)));

                static constexpr std::uint32_t Dim = simDim;

                HDINLINE TransformDataBox() = default;

                HDINLINE TransformDataBox(T_TranformFunctor transformFunc) : T_TranformFunctor(transformFunc)
                {
                }

                HDINLINE TransformDataBox(TransformDataBox const&) = default;

                HDINLINE ValueType operator()(DataSpace<simDim> const& idx) const
                {
                    return T_TranformFunctor::operator()(idx + m_offset);
                }

                HDINLINE ValueType operator[](DataSpace<simDim> const idx) const
                {
                    return T_TranformFunctor::operator()(idx + m_offset);
                }

                HDINLINE TransformDataBox shift(DataSpace<simDim> const& offset) const
                {
                    TransformDataBox result(*this);
                    result.m_offset += offset;
                    return result;
                }

                DataSpace<simDim> m_offset = DataSpace<simDim>::create(0);
            };

            auto Poisson::reduceGlobal(DataSpace<simDim> fieldSize, auto dataBoxIn)
            {
                DataBoxDim1Access d1Access(dataBoxIn, fieldSize);

                float_64 resultLocal
                    = (*localReduce)(pmacc::math::operation::Add(), d1Access, fieldSize.productOfComponents());

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                eventSystem::getTransactionEvent().waitForFinished();
                float_64 resultGlobal;
                mpiReduce(
                    pmacc::math::operation::Add(),
                    &resultGlobal,
                    &resultLocal,
                    1,
                    mpi::reduceMethods::AllReduce());

                return resultGlobal;
            }

            struct ForEachKernel
            {
                DINLINE auto operator()(auto const& worker, auto fieldOut, auto const func, auto const mapper) const
                    -> void
                {
                    DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));
                    DataSpace<simDim> superCellCellOffset = superCellIdx * SuperCellSize::toRT();

                    constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

                    auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);

                    forEachCellInSupercell(
                        [&](int32_t const linearCellIdx)
                        {
                            DataSpace<simDim> const cellIdx
                                = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
                            DataSpace<simDim> const dataCellOffset = superCellCellOffset + cellIdx;
                            fieldOut[dataCellOffset] = func(dataCellOffset);
                        });
                }
            };

            void Poisson::operator()(uint32_t const currentStep)
            {
                using namespace pmacc;
                constexpr uint fieldRhoSlot = 0;
                DataConnector& dc = Environment<>::get().DataConnector();
                auto& fieldRho = *dc.get<FieldTmp>(FieldTmp::getUniqueId(fieldRhoSlot));

                DataSpace<simDim> numGuardCells = fieldRho.getGridLayout().guardSizeND();
                DataSpace<simDim> coreBorderSize = fieldRho.getGridLayout().sizeWithoutGuardND();

                using EligibleSpecies = pmacc::mp_filter<SpeciesEligibleForChargeConservation, VectorAllSpecies>;
                /* calculate and add the charge density values from all species in FieldTmp */
                meta::ForEach<
                    EligibleSpecies,
                    detail::ComputeChargeDensity<boost::mpl::_1, pmacc::mp_int<CORE + BORDER>>,
                    boost::mpl::_1>
                    computeChargeDensity;

                fieldRho.getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));
                computeChargeDensity(fieldRho, currentStep);

                /* add results of all species that are still in GUARD to next GPUs BORDER */
                EventTask fieldTmpEvent = fieldRho.asyncCommunication(eventSystem::getTransactionEvent());
                eventSystem::setTransactionEvent(fieldTmpEvent);

                auto boundaryConditionsDirichlet = fields::poissonSolver::BoundaryConditionsDirichlet{};
                boundaryConditionsDirichlet(*fieldV.get(), m_mappingDesc);

                auto rightHandSideNormalization = fields::poissonSolver::RightHandSideNormalization{};
                rightHandSideNormalization(*fieldV.get(), fieldRho, m_mappingDesc);


                float_64 normRho;
                {
                    auto rhoDeviceBox = fieldRho.getDeviceDataBox().shift(numGuardCells);
                    TransformDataBox fieldTransform(
                        [rhoDeviceBox] DEVICEONLY(DataSpace<simDim> const& idx) -> float_64
                        { return precisionCast<float_64>(rhoDeviceBox[idx].x() * rhoDeviceBox[idx].x()); });

                    normRho = std::sqrt(reduceGlobal(coreBorderSize, fieldTransform));
                }
                // recalculate rho
                fieldRho.getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));
                computeChargeDensity(fieldRho, currentStep);
                /* add results of all species that are still in GUARD to next GPUs BORDER */
                eventSystem::setTransactionEvent(fieldRho.asyncCommunication(eventSystem::getTransactionEvent()));

                // normalize rho
                auto coreBorderMapper = makeAreaMapper<CORE + BORDER>(m_mappingDesc);
                {
                    auto rhoBox = fieldRho.getDeviceDataBox();

                    PMACC_LOCKSTEP_KERNEL(ForEachKernel{})
                        .config(coreBorderMapper.getGridDim(), SuperCellSize{})(
                            rhoBox,
                            DeviceLambda{
                                [rhoBox, normRho] DEVICEONLY(DataSpace<simDim> idx) -> float_64
                                { return rhoBox[idx].x() / normRho; }},
                            coreBorderMapper);
                }

                {
                    // normalize v
                    auto vMapper = makeAreaMapper<GUARD>(m_mappingDesc);
                    auto vField = fieldV->fieldVBuffer->getDeviceBuffer().getDataBox();
                    PMACC_LOCKSTEP_KERNEL(ForEachKernel{})
                        .config(vMapper.getGridDim(), SuperCellSize{})(
                            vField,
                            DeviceLambda{
                                [vField, normRho] DEVICEONLY(DataSpace<simDim> idx) -> float_64
                                { return vField[idx] / normRho; }},
                            vMapper);
                }

                {
                    auto vField = fieldV->fieldVBuffer->getDeviceBuffer().getDataBox();
                    auto r0Box = r0Buffer->getDeviceBuffer().getDataBox();
                    PMACC_LOCKSTEP_KERNEL(fields::poissonSolver::Stencil{})
                        .config(
                            coreBorderMapper.getGridDim(),
                            SuperCellSize{})(coreBorderMapper, fields::poissonSolver::StencilFunc{}, r0Box, vField);


                    auto rhoBox = fieldRho.getDeviceDataBox();
                    PMACC_LOCKSTEP_KERNEL(ForEachKernel{})
                        .config(coreBorderMapper.getGridDim(), SuperCellSize{})(
                            r0Box,
                            DeviceLambda{
                                [r0Box, rhoBox] DEVICEONLY(DataSpace<simDim> idx) -> float_64
                                { return rhoBox[idx].x() - r0Box[idx]; }},
                            coreBorderMapper);
                }

                pkBuffer->getDeviceBuffer().copyFrom(r0Buffer->getDeviceBuffer());
                rkBuffer->getDeviceBuffer().copyFrom(r0Buffer->getDeviceBuffer());

                float_64 rho0;
                /* rho reduction */
                {
                    auto r0Box = r0Buffer->getDeviceBuffer().getDataBox();
                    auto r0BoxBorderGuard = r0Box.shift(numGuardCells);

                    TransformDataBox fieldTransform(
                        [r0BoxBorderGuard] DEVICEONLY(DataSpace<simDim> const& idx) -> float_64
                        { return r0BoxBorderGuard[idx] * r0BoxBorderGuard[idx]; });

                    rho0 = reduceGlobal(coreBorderSize, fieldTransform);
                }

                float_64 rho1 = rho0;

                constexpr int maxIterations = 2000;
                constexpr float_64 epsilon = 1e-8;
                for(int i = 0; i < maxIterations; ++i)
                {
                    // preconditioner
                    mpkBuffer->getDeviceBuffer().copyFrom(pkBuffer->getDeviceBuffer());
                    mpkBuffer->communication();

                    // w = Ap
                    {
                        auto mpkBox = mpkBuffer->getDeviceBuffer().getDataBox();
                        auto ampkBox = ampkBuffer->getDeviceBuffer().getDataBox();
                        PMACC_LOCKSTEP_KERNEL(fields::poissonSolver::Stencil{})
                            .config(coreBorderMapper.getGridDim(), SuperCellSize{})(
                                coreBorderMapper,
                                fields::poissonSolver::StencilFunc{},
                                ampkBox,
                                mpkBox);
                    }

                    float_64 totalSum1;
                    /* local p = rw */
                    {
                        auto r0Box = r0Buffer->getDeviceBuffer().getDataBox();
                        auto r0BoxBorderGuard = r0Box.shift(numGuardCells);

                        auto ampkBox = ampkBuffer->getDeviceBuffer().getDataBox();
                        auto ampkBoxBorderGuard = ampkBox.shift(numGuardCells);

                        TransformDataBox fieldTransform(
                            [r0BoxBorderGuard, ampkBoxBorderGuard] DEVICEONLY(DataSpace<simDim> const& idx) -> float_64
                            { return r0BoxBorderGuard[idx] * ampkBoxBorderGuard[idx]; });

                        totalSum1 = reduceGlobal(coreBorderSize, fieldTransform);
                    }
                    float_64 alpha = rho0 / totalSum1;
                    // r = r - alpha * w
                    {
                        auto rkBox = rkBuffer->getDeviceBuffer().getDataBox();
                        auto ampkBox = ampkBuffer->getDeviceBuffer().getDataBox();

                        auto rhoBox = fieldRho.getDeviceDataBox();
                        PMACC_LOCKSTEP_KERNEL(ForEachKernel{})
                            .config(coreBorderMapper.getGridDim(), SuperCellSize{})(
                                rkBox,
                                DeviceLambda{
                                    [rkBox, ampkBox, alpha] DEVICEONLY(DataSpace<simDim> idx) -> float_64
                                    { return rkBox[idx] - alpha * ampkBox[idx]; }},
                                coreBorderMapper);
                    }

                    // preconditioner
                    zkBuffer->getDeviceBuffer().copyFrom(rkBuffer->getDeviceBuffer());
                    zkBuffer->communication();

                    // t = A * r
                    {
                        auto azkBox = azkBuffer->getDeviceBuffer().getDataBox();
                        auto zkBox = zkBuffer->getDeviceBuffer().getDataBox();
                        PMACC_LOCKSTEP_KERNEL(fields::poissonSolver::Stencil{})
                            .config(coreBorderMapper.getGridDim(), SuperCellSize{})(
                                coreBorderMapper,
                                fields::poissonSolver::StencilFunc{},
                                azkBox,
                                zkBox);
                    }

                    /* totalSum1 = azk * rk */
                    {
                        auto azkBox = azkBuffer->getDeviceBuffer().getDataBox();
                        auto azkBoxBorderGuard = azkBox.shift(numGuardCells);

                        auto rkBox = rkBuffer->getDeviceBuffer().getDataBox();
                        auto rkBoxBorderGuard = rkBox.shift(numGuardCells);

                        TransformDataBox fieldTransform(
                            [azkBoxBorderGuard, rkBoxBorderGuard] DEVICEONLY(DataSpace<simDim> const& idx) -> float_64
                            { return azkBoxBorderGuard[idx] * rkBoxBorderGuard[idx]; });

                        totalSum1 = reduceGlobal(coreBorderSize, fieldTransform);
                    }

                    float_64 totalSum2;
                    /* totalSum1 = azk * azk */
                    {
                        auto azkBox = azkBuffer->getDeviceBuffer().getDataBox();
                        auto azkBoxBorderGuard = azkBox.shift(numGuardCells);

                        TransformDataBox fieldTransform(
                            [azkBoxBorderGuard] DEVICEONLY(DataSpace<simDim> const& idx) -> float_64
                            { return azkBoxBorderGuard[idx] * azkBoxBorderGuard[idx]; });

                        totalSum2 = reduceGlobal(coreBorderSize, fieldTransform);
                    }

                    float_64 omega = totalSum1 / totalSum2;
                    // v = v + alpha * mpk + omega * zk
                    {
                        auto vFieldBox = fieldV->fieldVBuffer->getDeviceBuffer().getDataBox();
                        auto mpkBox = mpkBuffer->getDeviceBuffer().getDataBox();
                        auto zkBox = zkBuffer->getDeviceBuffer().getDataBox();

                        auto rhoBox = fieldRho.getDeviceDataBox();
                        PMACC_LOCKSTEP_KERNEL(ForEachKernel{})
                            .config(coreBorderMapper.getGridDim(), SuperCellSize{})(
                                vFieldBox,
                                DeviceLambda{
                                    [vFieldBox, mpkBox, zkBox, alpha, omega] DEVICEONLY(
                                        DataSpace<simDim> idx) -> float_64
                                    { return vFieldBox[idx] + alpha * mpkBox[idx] + omega * zkBox[idx]; }},
                                coreBorderMapper);
                    }
                    // rk = rk -  omega * azk
                    {
                        auto rkBox = rkBuffer->getDeviceBuffer().getDataBox();
                        auto azkBox = azkBuffer->getDeviceBuffer().getDataBox();


                        auto rhoBox = fieldRho.getDeviceDataBox();
                        PMACC_LOCKSTEP_KERNEL(ForEachKernel{})
                            .config(coreBorderMapper.getGridDim(), SuperCellSize{})(
                                rkBox,
                                DeviceLambda{
                                    [rkBox, azkBox, omega] DEVICEONLY(DataSpace<simDim> idx) -> float_64
                                    { return rkBox[idx] - omega * azkBox[idx]; }},
                                coreBorderMapper);
                    }

                    /* totalSum1 = r0 * rk */
                    {
                        auto r0Box = r0Buffer->getDeviceBuffer().getDataBox();
                        auto r0BoxBorderGuard = r0Box.shift(numGuardCells);

                        auto rkBox = rkBuffer->getDeviceBuffer().getDataBox();
                        auto rkBoxBorderGuard = rkBox.shift(numGuardCells);

                        TransformDataBox fieldTransform(
                            [r0BoxBorderGuard, rkBoxBorderGuard] DEVICEONLY(DataSpace<simDim> const& idx) -> float_64
                            { return r0BoxBorderGuard[idx] * rkBoxBorderGuard[idx]; });

                        totalSum1 = reduceGlobal(coreBorderSize, fieldTransform);
                    }
                    /* totalSum2 = rk * rk */
                    {
                        auto rkBox = rkBuffer->getDeviceBuffer().getDataBox();
                        auto rkBoxBorderGuard = rkBox.shift(numGuardCells);

                        TransformDataBox fieldTransform(
                            [rkBoxBorderGuard] DEVICEONLY(DataSpace<simDim> const& idx) -> float_64
                            { return rkBoxBorderGuard[idx] * rkBoxBorderGuard[idx]; });

                        totalSum2 = reduceGlobal(coreBorderSize, fieldTransform);
                    }

                    rho1 = totalSum1;
                    float_64 beta = rho1 / rho0 * alpha / omega;
                    rho0 = rho1;
                    if(std::sqrt(totalSum2) < epsilon)
                    {
                        std::cout << "Converged after " << i << " iterations with norm=" << normRho
                                  << ", total sum2=" << std::sqrt(totalSum2) << std::endl;
                        break;
                    }
                    // pk = rk + beta * (pk - omega * ampk)
                    {
                        auto pkBox = pkBuffer->getDeviceBuffer().getDataBox();
                        auto rkBox = rkBuffer->getDeviceBuffer().getDataBox();
                        auto ampkBox = ampkBuffer->getDeviceBuffer().getDataBox();

                        PMACC_LOCKSTEP_KERNEL(ForEachKernel{})
                            .config(coreBorderMapper.getGridDim(), SuperCellSize{})(
                                pkBox,
                                DeviceLambda{
                                    [pkBox, rkBox, ampkBox, beta, omega] DEVICEONLY(DataSpace<simDim> idx) -> float_64
                                    { return rkBox[idx] + beta * (pkBox[idx] - omega * ampkBox[idx]); }},
                                coreBorderMapper);
                    }
                } // for loop

                {
                    // normalize v back
                    auto vField = fieldV->fieldVBuffer->getDeviceBuffer().getDataBox();
                    PMACC_LOCKSTEP_KERNEL(ForEachKernel{})
                        .config(coreBorderMapper.getGridDim(), SuperCellSize{})(
                            vField,
                            DeviceLambda{
                                [vField, normRho] DEVICEONLY(DataSpace<simDim> idx) -> float_64
                                { return vField[idx] * normRho; }},
                            coreBorderMapper);
                    fieldV->fieldVBuffer->communication();

                    // compute fieldE
                    auto& eField = *dc.get<FieldE>(FieldE::getName());
                    auto fieldEBox = eField.getDeviceDataBox();
                    PMACC_LOCKSTEP_KERNEL(fields::poissonSolver::Stencil{})
                        .config(coreBorderMapper.getGridDim(), SuperCellSize{})(
                            coreBorderMapper,
                            fields::poissonSolver::GetFieldEStencil{},
                            fieldEBox,
                            vField);
                    eField.asyncCommunication(eventSystem::getTransactionEvent());
                    eventSystem::getTransactionEvent().waitForFinished();
                }
            }
        } // namespace stage
    } // namespace simulation
} // namespace picongpu
