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

#include <chrono>
#include <cstdint>
#include <tuple>

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

            Poisson::Poisson() : localReduce{std::make_unique<pmacc::device::Reduce>(1024)}
            {
            }

            void Poisson::registerHelp(po::options_description& desc)
            {
                namespace po = boost::program_options;
                po::options_description solverDesc("Poisson solver");

                solverDesc.add_options()(
                    "poisson.activate",
                    po::value<bool>(&m_useSolver)->zero_tokens(),
                    "enable poisson solver");
                solverDesc.add_options()(
                    "poisson.maxSteps",
                    po::value<uint32_t>(&m_maxSolverSteps)->default_value(2000),
                    "maximum number of steps for the preconditioner");
                solverDesc.add_options()(
                    "poisson.epsilon",
                    po::value<float_64>(&m_solverEpsilon)->default_value(1.0e-8),
                    "maximal allowed error of the poisson solver");
                // preconitioner
                solverDesc.add_options()(
                    "poisson.preconditioner.disable",
                    po::value<bool>(&m_disablePreconditioner)->zero_tokens(),
                    "disable poisson solver preconditioner");
                solverDesc.add_options()(
                    "poisson.preconditioner.maxSteps",
                    po::value<uint32_t>(&m_maxPreconditionerSteps)->default_value(20),
                    "maximum number of steps for the preconditioner");
                desc.add(solverDesc);
            }

            void Poisson::init(MappingDesc const mappingDesc)
            {
                m_mappingDesc = std::make_optional<MappingDesc>(mappingDesc);

                if(m_useSolver)
                {
                    pkBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc->getGridLayout());
                    rkBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc->getGridLayout());
                    r0Buffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc->getGridLayout());
                    mpkBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc->getGridLayout());
                    ampkBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc->getGridLayout());
                    zkBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc->getGridLayout());
                    azkBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc->getGridLayout());
                    fieldV = std::make_shared<fields::poissonSolver::FieldV>(m_mappingDesc.value());

                    yBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc->getGridLayout());
                    wBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc->getGridLayout());
                    zBuffer = std::make_unique<GridBuffer<float_64, simDim>>(m_mappingDesc->getGridLayout());

                    auto const commTag0 = pmacc::traits::getUniqueId<uint32_t>();
                    auto const commTag1 = pmacc::traits::getUniqueId<uint32_t>();
                    auto const commTagPk = pmacc::traits::getUniqueId<uint32_t>();
                    auto const commTagRk = pmacc::traits::getUniqueId<uint32_t>();
                    auto const commTagFieldV = pmacc::traits::getUniqueId<uint32_t>();

                    auto const commTagY = pmacc::traits::getUniqueId<uint32_t>();
                    /*go over all directions*/
                    for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
                    {
                        // set communication only for planes
                        if(FRONT % i == 0)
                        {
                            DataSpace<simDim> relativeMask = Mask::getRelativeDirections<simDim>(i);
                            /* guarding cells depend on direction
                             * for negative direction use originGuard else endGuard (relative direction ZERO is
                             * ignored) don't switch end and origin because this is a read buffer and no send buffer
                             */
                            auto guardingCells = DataSpace<simDim>::create(0);
                            for(uint32_t d = 0; d < simDim; ++d)
                                guardingCells[d] = (relativeMask[d] == 0 ? 0 : 1);
                            mpkBuffer->addExchange(GUARD, i, guardingCells, commTag0);
                            zkBuffer->addExchange(GUARD, i, guardingCells, commTag1);
                            pkBuffer->addExchange(GUARD, i, guardingCells, commTagPk);
                            rkBuffer->addExchange(GUARD, i, guardingCells, commTagRk);
                            fieldV->fieldVBuffer->addExchange(GUARD, i, guardingCells, commTagFieldV);

                            yBuffer->addExchange(GUARD, i, guardingCells, commTagY);
                        }
                    }
                    DataConnector& dc = Environment<>::get().DataConnector();
                    dc.share(fieldV);

                    participate(true);
                }
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

            template<typename T_TranformFunctor>
            inline auto Poisson::reduceGlobal(
                DataSpace<simDim> fieldSize,
                auto dataBoxIn,
                T_TranformFunctor reduceFunctor)
            {
                DataBoxDim1Access d1Access(dataBoxIn, fieldSize);

                float_64 resultLocal = (*localReduce)(reduceFunctor, d1Access, fieldSize.productOfComponents());

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                eventSystem::getTransactionEvent().waitForFinished();
                float_64 resultGlobal;
                mpiReduce(reduceFunctor, &resultGlobal, &resultLocal, 1, mpi::reduceMethods::AllReduce());

                return resultGlobal;
            }

            struct ForEachKernel
            {
                DINLINE auto operator()(
                    auto const& worker,
                    auto const& mapper,
                    auto const& func,
                    auto outBox,
                    auto... boxes) const -> void
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
                            outBox[dataCellOffset] = func(outBox[dataCellOffset], boxes[dataCellOffset]...);
                        });
                }
            };

            inline void forEachCell(auto mapper, auto const& functor, auto& outBuffer, auto&... buffers)
            {
                PMACC_LOCKSTEP_KERNEL(ForEachKernel{})
                    .config(mapper.getGridDim(), SuperCellSize{})(
                        mapper,
                        DeviceLambda{functor},
                        outBuffer.getDataBox(),
                        buffers.getDataBox()...);
            }

            void Poisson::preconditioner(
                std::unique_ptr<GridBuffer<float_64, simDim>>& xBuffer,
                std::unique_ptr<GridBuffer<float_64, simDim>>& bBuffer)
            {
                yBuffer->getDeviceBuffer().setValue(0.0);
                wBuffer->getDeviceBuffer().setValue(0.0);
                zBuffer->getDeviceBuffer().setValue(0.0);

                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                auto globalDomain = subGrid.getGlobalDomain().size;
                auto cellSizeSquared = sim.pic.getCellSize<float_64>() * sim.pic.getCellSize<float_64>();

                float_64 eigenMin = 0.0;
                float_64 eigenMax = 0.0;

                for(uint32_t d = 0; d < simDim; ++d)
                {
                    eigenMin += 4.0 * math::sin(1.0 * pmacc::math::Pi<float_64>::halfValue / (globalDomain[d] + 1))
                                * math::sin(1.0 * pmacc::math::Pi<float_64>::halfValue / (globalDomain[d] + 1))
                                / (cellSizeSquared[d]);

                    eigenMax
                        += 4.0
                           * math::sin(globalDomain[d] * pmacc::math::Pi<float_64>::halfValue / (globalDomain[d] + 1))
                           * math::sin(globalDomain[d] * pmacc::math::Pi<float_64>::halfValue / (globalDomain[d] + 1))
                           / (cellSizeSquared[d]);
                }

                float_64 const theta = 0.5 * (eigenMax + eigenMin);
                float_64 const delta = 0.5 * (eigenMax - eigenMin);
                float_64 const sigma = theta / delta;

                float_64 rhoOld = 1. / sigma;
                float_64 rhoCurrent = 1. / (2. * sigma - rhoOld);

                bBuffer->communication();

                auto coreBorderMapper = makeAreaMapper<CORE + BORDER>(m_mappingDesc.value());

                namespace poi = fields::poissonSolver;

                forEachCell(
                    coreBorderMapper,
                    [theta] DEVICEONLY(auto, auto const bValue) -> float_64 { return bValue / theta; },
                    zBuffer->getDeviceBuffer(),
                    bBuffer->getDeviceBuffer());

                poi::stencil(
                    coreBorderMapper,
                    poi::StencilFunc{},
                    yBuffer->getDeviceBuffer(),
                    bBuffer->getDeviceBuffer());

                forEachCell(
                    coreBorderMapper,
                    [rhoCurrent, theta, delta] DEVICEONLY(auto const yValue) -> float_64
                    { return -2.0 * rhoCurrent * yValue / theta / delta; },
                    yBuffer->getDeviceBuffer());

                forEachCell(
                    coreBorderMapper,
                    [rhoCurrent, delta] DEVICEONLY(auto const yValue, auto const bValue) -> float_64
                    { return yValue + 4.0 * bValue * rhoCurrent / delta; },
                    yBuffer->getDeviceBuffer(),
                    bBuffer->getDeviceBuffer());


                uint32_t iterMax = m_maxPreconditionerSteps;
                for(uint32_t i = 2; i < iterMax; ++i)
                {
                    rhoOld = rhoCurrent;
                    rhoCurrent = 1. / (2. * sigma - rhoOld);
                    yBuffer->communication();

                    poi::stencil(
                        coreBorderMapper,
                        poi::StencilFunc{},
                        wBuffer->getDeviceBuffer(),
                        yBuffer->getDeviceBuffer());

                    forEachCell(
                        coreBorderMapper,
                        [rhoCurrent, delta] DEVICEONLY(auto const wValue) -> float_64
                        { return -2.0 * rhoCurrent * wValue / delta; },
                        wBuffer->getDeviceBuffer());

                    forEachCell(
                        coreBorderMapper,
                        [rhoCurrent, delta] DEVICEONLY(auto const wValue, auto const bValue) -> float_64
                        { return wValue + 2.0 * rhoCurrent * bValue / delta; },
                        wBuffer->getDeviceBuffer(),
                        bBuffer->getDeviceBuffer());

                    forEachCell(
                        coreBorderMapper,
                        [rhoCurrent, sigma] DEVICEONLY(auto const wValue, auto const yValue) -> float_64
                        { return wValue + 2.0 * rhoCurrent * sigma * yValue; },
                        wBuffer->getDeviceBuffer(),
                        yBuffer->getDeviceBuffer());

                    forEachCell(
                        coreBorderMapper,
                        [rhoCurrent, rhoOld] DEVICEONLY(auto const wValue, auto const zValue) -> float_64
                        { return wValue - rhoCurrent * rhoOld * zValue; },
                        wBuffer->getDeviceBuffer(),
                        zBuffer->getDeviceBuffer());

                    zBuffer->getDeviceBuffer().copyFrom(yBuffer->getDeviceBuffer());
                    yBuffer->getDeviceBuffer().copyFrom(wBuffer->getDeviceBuffer());
                } // loop
                xBuffer->getDeviceBuffer().copyFrom(wBuffer->getDeviceBuffer());
            }

            void Poisson::operator()(uint32_t const currentStep)
            {
                namespace poi = fields::poissonSolver;

                // only one rank is writing status initial information of the poisson solver
                bool mainRank = Environment<simDim>::get().GridController().getGlobalRank() == 0;
                if(mainRank)
                    log<picLog::PHYSICS>("Poisson solver:");
                if(!m_useSolver)
                {
                    if(mainRank)
                        log<picLog::PHYSICS>("  - disabled");
                    return;
                }
                eventSystem::getTransactionEvent().waitForFinished();
                auto beginT = std::chrono::high_resolution_clock::now();

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

                auto boundaryConditionsDirichlet = poi::BoundaryConditionsDirichlet{};
                boundaryConditionsDirichlet(*fieldV.get(), m_mappingDesc.value());

                auto rightHandSideNormalization = poi::RightHandSideNormalization{};
                rightHandSideNormalization(*fieldV.get(), fieldRho, m_mappingDesc.value());


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

                auto coreBorderMapper = makeAreaMapper<CORE + BORDER>(m_mappingDesc.value());

                // normalize rho
                forEachCell(
                    coreBorderMapper,
                    [normRho] DEVICEONLY(auto rhoValue) -> float_64 { return rhoValue.x() / normRho; },
                    fieldRho.getGridBuffer().getDeviceBuffer());

                auto vMapper = makeAreaMapper<GUARD>(m_mappingDesc.value());

                // normalize v
                forEachCell(
                    vMapper,
                    [normRho] DEVICEONLY(auto const vValue) -> float_64 { return vValue / normRho; },
                    fieldV->fieldVBuffer->getDeviceBuffer());

                poi::stencil(
                    coreBorderMapper,
                    poi::StencilFunc{},
                    r0Buffer->getDeviceBuffer(),
                    fieldV->fieldVBuffer->getDeviceBuffer());

                forEachCell(
                    coreBorderMapper,
                    [] DEVICEONLY(auto const r0Value, auto const rhoValue) -> float_64
                    { return rhoValue.x() - r0Value; },
                    r0Buffer->getDeviceBuffer(),
                    fieldRho.getGridBuffer().getDeviceBuffer());

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
                float_64 totalSum2;

                int maxIterations = m_maxSolverSteps;

                bool foundSolution = false;
                int iteration = 0;
                for(; iteration < maxIterations; ++iteration)
                {
                    // preconditioner
                    if(m_disablePreconditioner)
                        mpkBuffer->getDeviceBuffer().copyFrom(pkBuffer->getDeviceBuffer());
                    else
                        preconditioner(mpkBuffer, pkBuffer);

                    mpkBuffer->communication();

                    // w = Ap
                    poi::stencil(
                        coreBorderMapper,
                        poi::StencilFunc{},
                        ampkBuffer->getDeviceBuffer(),
                        mpkBuffer->getDeviceBuffer());

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
                    forEachCell(
                        coreBorderMapper,
                        [alpha] DEVICEONLY(auto const rkValue, auto const ampkValue) -> float_64
                        { return rkValue - alpha * ampkValue; },
                        rkBuffer->getDeviceBuffer(),
                        ampkBuffer->getDeviceBuffer());

                    // preconditioner
                    if(m_disablePreconditioner)
                        zkBuffer->getDeviceBuffer().copyFrom(rkBuffer->getDeviceBuffer());
                    else
                        preconditioner(zkBuffer, rkBuffer);

                    zkBuffer->communication();

                    // t = A * r
                    poi::stencil(
                        coreBorderMapper,
                        poi::StencilFunc{},
                        azkBuffer->getDeviceBuffer(),
                        zkBuffer->getDeviceBuffer());

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
                    forEachCell(
                        coreBorderMapper,
                        [alpha,
                         omega] DEVICEONLY(auto const vValue, auto const mpkValue, auto const zkValue) -> float_64
                        { return vValue + alpha * mpkValue + omega * zkValue; },
                        fieldV->fieldVBuffer->getDeviceBuffer(),
                        mpkBuffer->getDeviceBuffer(),
                        zkBuffer->getDeviceBuffer());

                    // rk = rk -  omega * azk
                    forEachCell(
                        coreBorderMapper,
                        [omega] DEVICEONLY(auto const rkValue, auto const azkValue) -> float_64
                        { return rkValue - omega * azkValue; },
                        rkBuffer->getDeviceBuffer(),
                        azkBuffer->getDeviceBuffer());

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
                    if(std::sqrt(totalSum2) < m_solverEpsilon)
                    {
                        foundSolution = true;
                        break;
                    }
                    // pk = rk + beta * (pk - omega * ampk)
                    forEachCell(
                        coreBorderMapper,
                        [beta,
                         omega] DEVICEONLY(auto const pkValue, auto const rkValue, auto const ampkValue) -> float_64
                        { return rkValue + beta * (pkValue - omega * ampkValue); },
                        pkBuffer->getDeviceBuffer(),
                        rkBuffer->getDeviceBuffer(),
                        ampkBuffer->getDeviceBuffer());

                } // for loop

                // avid deadlock due blocking colective MPI operation
                eventSystem::getTransactionEvent().waitForFinished();

                bool ioRank = mpiReduce.hasResult(mpi::reduceMethods::Reduce());

                int maxGlobalIterations = 0;
                mpiReduce(
                    pmacc::math::operation::Max(),
                    &maxGlobalIterations,
                    &iteration,
                    1,
                    mpi::reduceMethods::Reduce());

                float_64 maxGlobalNormRho = 0.0;
                mpiReduce(pmacc::math::operation::Max(), &maxGlobalNormRho, &normRho, 1, mpi::reduceMethods::Reduce());

                float_64 maxGlobalTotalSum2 = 0.0;
                mpiReduce(
                    pmacc::math::operation::Max(),
                    &maxGlobalTotalSum2,
                    &totalSum2,
                    1,
                    mpi::reduceMethods::Reduce());

                if(foundSolution)
                {
                    if(ioRank)
                        log<picLog::PHYSICS>("  - converged after %1%/%2% iterations with norm=%3%, total epsilon=%4%")
                            % maxGlobalIterations % maxIterations % maxGlobalNormRho % std::sqrt(maxGlobalTotalSum2);

                    // normalize v back
                    forEachCell(
                        coreBorderMapper,
                        [normRho] DEVICEONLY(auto const vValue) -> float_64
                        { return vValue * normRho / sim.pic.getEps0(); },
                        fieldV->fieldVBuffer->getDeviceBuffer());
                    fieldV->fieldVBuffer->communication();

                    // compute fieldE
                    auto& eField = *dc.get<FieldE>(FieldE::getName());

                    poi::stencil(
                        coreBorderMapper,
                        poi::GetFieldEStencil{},
                        eField.getGridBuffer().getDeviceBuffer(),
                        fieldV->fieldVBuffer->getDeviceBuffer());
                    setTransactionEvent(eField.asyncCommunication(eventSystem::getTransactionEvent()));
                }

                eventSystem::getTransactionEvent().waitForFinished();
                auto endT = std::chrono::high_resolution_clock::now();
                double duration = std::chrono::duration<double>(endT - beginT).count();

                // avid deadlock due blocking collective MPI operation
                eventSystem::getTransactionEvent().waitForFinished();
                double globalMaxDuration = 0.0;
                mpiReduce(
                    pmacc::math::operation::Max(),
                    &globalMaxDuration,
                    &duration,
                    1,
                    mpi::reduceMethods::Reduce());
                if(ioRank)
                    log<picLog::PHYSICS>("  - duration %1% sec") % globalMaxDuration;

                if(ioRank && !foundSolution)
                {
                    log<picLog::PHYSICS>("  - not converge after %1% iterations with norm=%2%, total epsilon=%3%")
                        % maxIterations % normRho % std::sqrt(rho1);
                    throw std::runtime_error("Poisson solver did not converge after max iterations");
                }
            }
        } // namespace stage
    } // namespace simulation
} // namespace picongpu
