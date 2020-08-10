/* Copyright 2019 Benjamin Worpitz
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
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/queue/Traits.hpp>

#include <string>
#include <typeinfo>
#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The accelerator specifics.
    namespace acc
    {
        struct ConceptUniformCudaHip{};

        struct ConceptAcc{};
        //-----------------------------------------------------------------------------
        //! The accelerator traits.
        namespace traits
        {
            //#############################################################################
            //! The accelerator type trait.
            template<
                typename T,
                typename TSfinae = void>
            struct AccType;

            //#############################################################################
            //! The device properties get trait.
            template<
                typename TAcc,
                typename TSfinae = void>
            struct GetAccDevProps;

            //#############################################################################
            //! The accelerator name trait.
            //!
            //! The default implementation returns the mangled class name.
            template<
                typename TAcc,
                typename TSfinae = void>
            struct GetAccName
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return typeid(TAcc).name();
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator device properties get trait specialization.
            template<typename TAcc>
            struct GetAccDevProps<
                TAcc,
                typename std::enable_if<
                    concepts::ImplementsConcept<acc::ConceptUniformCudaHip, TAcc>::value
                >::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    typename dev::traits::DevType<TAcc>::type const & dev)
                -> AccDevProps<typename dim::traits::DimType<TAcc>::type, typename idx::traits::IdxType<TAcc>::type>
                {
                    using ImplementationBase = typename concepts::ImplementationBase<acc::ConceptUniformCudaHip, TAcc>;
                    return GetAccDevProps<ImplementationBase>::getAccDevProps(dev);
                }
            };
        }

        //#############################################################################
        //! The accelerator type trait alias template to remove the ::type.
        template<
            typename T>
        using Acc = typename traits::AccType<T>::type;

        //-----------------------------------------------------------------------------
        //! \return The acceleration properties on the given device.
        template<
            typename TAcc,
            typename TDev>
        ALPAKA_FN_HOST auto getAccDevProps(
            TDev const & dev)
        -> AccDevProps<dim::Dim<TAcc>, idx::Idx<TAcc>>
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptAcc, TAcc>;
            return
                traits::GetAccDevProps<
                    ImplementationBase>
                ::getAccDevProps(
                    dev);
        }

        //-----------------------------------------------------------------------------
        //! \return The accelerator name
        //!
        //! \tparam TAcc The accelerator type.
        template<
            typename TAcc>
        ALPAKA_FN_HOST auto getAccName()
        -> std::string
        {
            return
                traits::GetAccName<
                    TAcc>
                ::getAccName();
        }
    }

    namespace kernel
    {
        namespace detail
        {
            template<typename TAcc>
            struct CheckFnReturnType<
                TAcc,
                typename std::enable_if<
                    concepts::ImplementsConcept<acc::ConceptUniformCudaHip, TAcc>::value
                >::type>
            {
                 template<
                    typename TKernelFnObj,
                    typename... TArgs>
                void operator()(
                    TKernelFnObj const & kernelFnObj,
                    TArgs const & ... args)
                {
                    using ImplementationBase = typename concepts::ImplementationBase<acc::ConceptUniformCudaHip, TAcc>;
                    CheckFnReturnType<ImplementationBase>{}(
                        kernelFnObj,
                        args...);
                }
            };
        }

    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator device type trait specialization.
            template<typename TAcc>
            struct DevType<
               TAcc,
                typename std::enable_if<
                    concepts::ImplementsConcept<acc::ConceptUniformCudaHip, TAcc>::value
                >::type>
            {
                using ImplementationBase = typename concepts::ImplementationBase<acc::ConceptUniformCudaHip, TAcc>;
                using type = typename DevType<ImplementationBase>::type;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU HIP execution task platform type trait specialization.
            template<typename TAcc>
            struct PltfType<
                TAcc,
                typename std::enable_if<
                    concepts::ImplementsConcept<acc::ConceptUniformCudaHip, TAcc>::value
                >::type>
                {
                    using ImplementationBase = typename concepts::ImplementationBase<acc::ConceptUniformCudaHip, TAcc>;
                    using type = typename PltfType<ImplementationBase>::type;
                };
        }

    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator dimension getter trait specialization.
            template<typename TAcc>
            struct DimType<
                TAcc,
                typename std::enable_if<
                    concepts::ImplementsConcept<acc::ConceptUniformCudaHip, TAcc>::value
                >::type>
            {
                    using ImplementationBase = typename concepts::ImplementationBase<acc::ConceptUniformCudaHip, TAcc>;
                    using type = typename DimType<ImplementationBase>::type;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU HIP accelerator idx type trait specialization.
            template<typename TAcc>
            struct IdxType<
                TAcc,
                typename std::enable_if<
                    concepts::ImplementsConcept<acc::ConceptUniformCudaHip, TAcc>::value
                >::type>
            {
                using ImplementationBase = typename concepts::ImplementationBase<acc::ConceptUniformCudaHip, TAcc>;
                using type = typename IdxType<ImplementationBase>::type;
            };
        }
    }

    namespace queue
    {
        namespace traits
        {
            template<
                typename TAcc,
                typename TProperty>
            struct QueueType<
                TAcc,
                TProperty,
                std::enable_if_t<
                    concepts::ImplementsConcept<acc::ConceptAcc, TAcc>::value
                >
            >
            {
                using type = typename QueueType<
                    typename pltf::traits::PltfType<TAcc>::type,
                    TProperty
                >::type;
            };
        }
    }
}
