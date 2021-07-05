/* Copyright 2021 Pawel Ordyna
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/scattering/scattering.kernel"

#include <pmacc/traits/GetFlagType.hpp>

#include <functional>
#include <tuple>
#include <utility>

namespace picongpu
{
    namespace particles
    {
        namespace scattering
        {
            namespace detail
            {
                template<size_t i>
                struct GetFieldTmp
                {
                    template<typename T_Dc>
                    HINLINE std::shared_ptr<FieldTmp> operator()(T_Dc& dc)
                    {
                        return dc.template get<FieldTmp>(FieldTmp::getUniqueId(i), true);
                    }
                };
                template<typename T_Dc, std::size_t... I>
                HINLINE auto getFieldPointersTuple(T_Dc& dc, std::index_sequence<I...>)
                {
                    return std::make_tuple(GetFieldTmp<I>()(dc)...);
                }
                template<typename... T_Args, std::size_t... I>
                HINLINE auto getDataBoxTuple(std::tuple<T_Args...>& fieldPointers, std::index_sequence<I...>)
                {
                    return std::make_tuple(
                        (std::get<I>(fieldPointers)->getGridBuffer().getDeviceBuffer().getDataBox())...);
                }

                template<typename T_FieldTmpOperations, size_t i>
                struct ComputeFieldValuesImpl
                {
                    using FieldTmpOp = typename bmpl::at_c<T_FieldTmpOperations, i>::type;
                    using Species = typename FieldTmpOp::Species;
                    using Solver = typename FieldTmpOp::Solver;

                    template<typename T_Dc, typename... T_Args>
                    HINLINE void operator()(
                        T_Dc& dc,
                        uint32_t const& currentStep,
                        std::tuple<T_Args...>& fieldPointers)
                    {
                        auto species = dc.template get<Species>(Species::FrameType::getName(), true);
                        std::get<i>(fieldPointers)
                            ->template computeValue<CORE + BORDER, Solver>(*species, currentStep);
                    }
                };

                template<template<size_t> class T_Functor>
                struct CallComputeFieldValues
                {
                    template<typename T_Dc, typename... T_Args, std::size_t... I>
                    HINLINE void operator()(
                        T_Dc& dc,
                        uint32_t const& currentStep,
                        std::tuple<T_Args...> fieldPointers,
                        std::index_sequence<I...>)
                    {
                        int t[] = {((void) T_Functor<I>()(dc, currentStep, fieldPointers), 1)...};
                        (void) t;
                    }
                };


                struct CallKernel
                {
                    template<
                        typename T_Mapper,
                        typename T_ParticleBox,
                        typename T_HostScattererFunctor,
                        typename... T_FieldBoxes>
                    HINLINE void operator()(
                        T_Mapper const& mapper,
                        T_ParticleBox& particleBox,
                        T_HostScattererFunctor& hostScattererFunctor,
                        T_FieldBoxes&... fieldBoxes)
                    {
                        constexpr uint32_t numWorkers
                            = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;
                        PMACC_KERNEL(acc::ScatterParticlesKernel<numWorkers>{})
                        (mapper.getGridDim(), numWorkers)(mapper, particleBox, hostScattererFunctor, fieldBoxes...);
                    }
                };

                template<typename T_Func, typename... T_values, std::size_t... I, typename... T_Args>
                HINLINE void callWithTuple(
                    T_Func func,
                    const std::tuple<T_values...>& tuple,
                    std::index_sequence<I...>,
                    T_Args&&... args)
                {
                    func(args..., std::get<I>(tuple)...);
                }

            } // namespace detail

            template<typename T_SpeciesType>
            struct CallScatterer
            {
                using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
                using FrameType = typename SpeciesType::FrameType;

                using Scatterer =
                    typename pmacc::traits::Resolve<typename GetFlagType<FrameType, scatterer<>>::type>::type;
                using RequiredDerivedFields =
                    typename pmacc::traits::Resolve<typename Scatterer::RequiredDerivedFields>::type;
                template<size_t i>
                using ComputeFieldValues = detail::ComputeFieldValuesImpl<RequiredDerivedFields, i>;

                /** Functor implementation
                 *
                 * @tparam T_CellDescription contains the number of blocks and blocksize
                 *                           that is later passed to the kernel
                 * @param cellDesc logical block information like dimension and cell sizes
                 * @param currentStep The current time step
                 */
                HINLINE void operator()(const uint32_t currentStep) const
                {
                    /* List of fields that are required by the functor.
                     *
                     * This is an mpl sequence of already specialized FieldTmpOperations.
                     */


                    // Get access to tmp fields.
                    constexpr size_t numFields = bmpl::size<RequiredDerivedFields>::type::value;
                    PMACC_CASSERT_MSG(
                        _please_allocate_at_least_as_many_FieldTmp_slots_in_memory_param_as_fields_required_for_scattering,
                        fieldTmpNumSlots >= numFields);
                    DataConnector& dc = Environment<>::get().DataConnector();
                    std::make_index_sequence<numFields> index{};
                    auto fieldPointers = detail::getFieldPointersTuple(dc, index);

                    // TODO: Do I need to initalize it with zeros as well?
                    detail::CallComputeFieldValues<ComputeFieldValues>()(dc, currentStep, fieldPointers, index);
                    auto dataBoxes = detail::getDataBoxTuple(fieldPointers, index);

                    auto species = dc.get<SpeciesType>(FrameType::getName(), true);
                    AreaMapping<CORE + BORDER, picongpu::MappingDesc> mapper(species->getCellDescription());

                    detail::callWithTuple(detail::CallKernel(), dataBoxes, index, mapper,
                        species->getDeviceParticlesBox(),
                        Scatterer(currentStep) );
                }
            };
        } // namespace scattering
    } // namespace particles
} // namespace picongpu
