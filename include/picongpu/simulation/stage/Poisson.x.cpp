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
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/simulation/stage/Poisson.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#include <pmacc/memory/boxes/DataBoxUnaryTransform.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

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

            template<typename T>
            using SpeciesEligibleForChargeDeposition =
                typename particles::traits::SpeciesEligibleForSolver<T, simulation::stage::Poisson>::type;

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

                DataConnector& dc = Environment<>::get().DataConnector();
                dc.share(fieldV);

                participate(true);
            }

            template<typename T_Type>
            struct cast64Bit
            {
                using result = typename TypeCast<float_64, T_Type>::result;

                HDINLINE result operator()(T_Type const& value) const
                {
                    return precisionCast<float_64>(value);
                }
            };

            template<typename T_Type>
            struct squareComponentWise
            {
                using result = T_Type;

                HDINLINE result operator()(T_Type const& value) const
                {
                    return value * value;
                }
            };

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

            auto Poisson::calcNorm(FieldTmp& fieldRho)
            {
                /* reduce field E*/
                DataSpace<simDim> fieldSize = fieldRho.getGridLayout().sizeWithoutGuardND();
                DataSpace<simDim> fieldGuard = fieldRho.getGridLayout().guardSizeND();

                auto rhoDeviceBox = fieldRho.getDeviceDataBox().shift(fieldGuard);

                TransformDataBox fieldTransform(
                    [rhoDeviceBox] DEVICEONLY(DataSpace<simDim> const& idx)
                    { return precisionCast<float_64>(rhoDeviceBox[idx] * rhoDeviceBox[idx]); });
                DataBoxDim1Access d1Access(fieldTransform, fieldSize);

                float_64 fieldRhoNormSquaredLocal
                    = (*localReduce)(pmacc::math::operation::Add(), d1Access, fieldSize.productOfComponents()).x();

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                eventSystem::getTransactionEvent().waitForFinished();
                float_64 fieldRhoNormSquaredGlobal;
                mpiReduce(
                    pmacc::math::operation::Add(),
                    &fieldRhoNormSquaredGlobal,
                    &fieldRhoNormSquaredLocal,
                    1,
                    mpi::reduceMethods::AllReduce());

                return math::sqrt(fieldRhoNormSquaredGlobal);
            }

            struct NormalizeField
            {
                DINLINE auto operator()(auto const& worker, auto fieldBox, float_64 normValue, auto const mapper) const
                    -> void
                {
                    DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));
                    DataSpace<simDim> superCellTotalCellOffset = superCellIdx * SuperCellSize::toRT();

                    constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

                    auto forEachCellInSupercell = lockstep::makeForEach<cellsPerSuperCell>(worker);

                    forEachCellInSupercell(
                        [&](int32_t const linearCellIdx)
                        {
                            DataSpace<simDim> const cellIdx
                                = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
                            DataSpace<simDim> const dataCellOffset = superCellTotalCellOffset + cellIdx;
                            fieldBox(dataCellOffset) /= normValue;
                        });
                }
            };

            void Poisson::operator()(uint32_t const currentStep)
            {
                using namespace pmacc;
                constexpr uint fieldRhoSlot = 0;
                DataConnector& dc = Environment<>::get().DataConnector();
                auto& fieldRho = *dc.get<FieldTmp>(FieldTmp::getUniqueId(fieldRhoSlot));

                fieldRho.getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));


                using EligibleSpecies = pmacc::mp_filter<SpeciesEligibleForChargeDeposition, VectorAllSpecies>;

                // todo: log species that are used / ignored in this plugin with INFO


                /* calculate and add the charge density values from all species in FieldTmp */
                meta::ForEach<
                    EligibleSpecies,
                    detail::ComputeChargeDensity<boost::mpl::_1, pmacc::mp_int<CORE + BORDER>>,
                    boost::mpl::_1>
                    computeChargeDensity;
                computeChargeDensity(fieldRho, currentStep);

                /* add results of all species that are still in GUARD to next GPUs BORDER */
                EventTask fieldTmpEvent = fieldRho.asyncCommunication(eventSystem::getTransactionEvent());
                eventSystem::setTransactionEvent(fieldTmpEvent);

                auto boundaryConditionsDirichlet = fields::poissonSolver::BoundaryConditionsDirichlet{};
                boundaryConditionsDirichlet(*fieldV.get(), m_mappingDesc);

                auto rightHandSideNormalization = fields::poissonSolver::RightHandSideNormalization{};
                rightHandSideNormalization(*fieldV.get(), fieldRho, m_mappingDesc);
                float_64 normRho = calcNorm(fieldRho);

                // recalculate rho
                computeChargeDensity(fieldRho, currentStep);
                /* add results of all species that are still in GUARD to next GPUs BORDER */
                eventSystem::setTransactionEvent(fieldRho.asyncCommunication(eventSystem::getTransactionEvent()));

                // normalize rho
                auto rhoMapper = makeAreaMapper<CORE + BORDER>(m_mappingDesc);
                PMACC_LOCKSTEP_KERNEL(NormalizeField{})
                    .config(rhoMapper.getGridDim(), SuperCellSize{})(fieldRho.getDeviceDataBox(), normRho, rhoMapper);

                // normalize v
                auto vMapper = makeAreaMapper<GUARD>(m_mappingDesc);
                PMACC_LOCKSTEP_KERNEL(NormalizeField{})
                    .config(
                        vMapper.getGridDim(),
                        SuperCellSize{})(fieldV->fieldVBuffer->getDeviceBuffer().getDataBox(), normRho, rhoMapper);

                // BICGStab(fieldV, fieldRho, cellDescription);
            }
        } // namespace stage
    } // namespace simulation
} // namespace picongpu
