/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/queue/Traits.hpp>

#include <string>
#include <type_traits>
#include <typeinfo>

namespace alpaka
{
    struct ConceptUniformCudaHip
    {
    };

    struct ConceptAcc
    {
    };
    //! The accelerator traits.
    namespace trait
    {
        //! The accelerator type trait.
        template<typename T, typename TSfinae = void>
        struct AccType;

        //! The device properties get trait.
        template<typename TAcc, typename TSfinae = void>
        struct GetAccDevProps;

        //! The accelerator name trait.
        //!
        //! The default implementation returns the mangled class name.
        template<typename TAcc, typename TSfinae = void>
        struct GetAccName
        {
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return typeid(TAcc).name();
            }
        };

        //! The GPU CUDA accelerator device properties get trait specialization.
        template<typename TAcc>
        struct GetAccDevProps<TAcc, std::enable_if_t<concepts::ImplementsConcept<ConceptUniformCudaHip, TAcc>::value>>
        {
            ALPAKA_FN_HOST static auto getAccDevProps(typename alpaka::trait::DevType<TAcc>::type const& dev)
                -> AccDevProps<typename trait::DimType<TAcc>::type, typename trait::IdxType<TAcc>::type>
            {
                using ImplementationBase = typename concepts::ImplementationBase<ConceptUniformCudaHip, TAcc>;
                return GetAccDevProps<ImplementationBase>::getAccDevProps(dev);
            }
        };
    } // namespace trait

    //! The accelerator type trait alias template to remove the ::type.
    template<typename T>
    using Acc = typename trait::AccType<T>::type;

    //! \return The acceleration properties on the given device.
    template<typename TAcc, typename TDev>
    ALPAKA_FN_HOST auto getAccDevProps(TDev const& dev) -> AccDevProps<Dim<TAcc>, Idx<TAcc>>
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptAcc, TAcc>;
        return trait::GetAccDevProps<ImplementationBase>::getAccDevProps(dev);
    }

    //! \return The accelerator name
    //!
    //! \tparam TAcc The accelerator type.
    template<typename TAcc>
    ALPAKA_FN_HOST auto getAccName() -> std::string
    {
        return trait::GetAccName<TAcc>::getAccName();
    }

    namespace detail
    {
        template<typename TAcc>
        struct CheckFnReturnType<
            TAcc,
            std::enable_if_t<concepts::ImplementsConcept<ConceptUniformCudaHip, TAcc>::value>>
        {
            template<typename TKernelFnObj, typename... TArgs>
            void operator()(TKernelFnObj const& kernelFnObj, TArgs const&... args)
            {
                using ImplementationBase = typename concepts::ImplementationBase<ConceptUniformCudaHip, TAcc>;
                CheckFnReturnType<ImplementationBase>{}(kernelFnObj, args...);
            }
        };
    } // namespace detail

    namespace trait
    {
        //! The GPU HIP accelerator device type trait specialization.
        template<typename TAcc>
        struct DevType<TAcc, std::enable_if_t<concepts::ImplementsConcept<ConceptUniformCudaHip, TAcc>::value>>
        {
            using ImplementationBase = typename concepts::ImplementationBase<ConceptUniformCudaHip, TAcc>;
            using type = typename DevType<ImplementationBase>::type;
        };

        //! The CPU HIP execution task platform type trait specialization.
        template<typename TAcc>
        struct PltfType<TAcc, std::enable_if_t<concepts::ImplementsConcept<ConceptUniformCudaHip, TAcc>::value>>
        {
            using ImplementationBase = typename concepts::ImplementationBase<ConceptUniformCudaHip, TAcc>;
            using type = typename PltfType<ImplementationBase>::type;
        };

        //! The GPU HIP accelerator dimension getter trait specialization.
        template<typename TAcc>
        struct DimType<TAcc, std::enable_if_t<concepts::ImplementsConcept<ConceptUniformCudaHip, TAcc>::value>>
        {
            using ImplementationBase = typename concepts::ImplementationBase<ConceptUniformCudaHip, TAcc>;
            using type = typename DimType<ImplementationBase>::type;
        };

        //! The GPU HIP accelerator idx type trait specialization.
        template<typename TAcc>
        struct IdxType<TAcc, std::enable_if_t<concepts::ImplementsConcept<ConceptUniformCudaHip, TAcc>::value>>
        {
            using ImplementationBase = typename concepts::ImplementationBase<ConceptUniformCudaHip, TAcc>;
            using type = typename IdxType<ImplementationBase>::type;
        };

        template<typename TAcc, typename TProperty>
        struct QueueType<TAcc, TProperty, std::enable_if_t<concepts::ImplementsConcept<ConceptAcc, TAcc>::value>>
        {
            using type = typename QueueType<typename alpaka::trait::PltfType<TAcc>::type, TProperty>::type;
        };
    } // namespace trait
} // namespace alpaka
