/* Copyright 2024 Benjamin Worpitz, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccDevProps.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/platform/Traits.hpp"
#include "alpaka/queue/Traits.hpp"

#include <string>
#include <type_traits>
#include <typeinfo>

namespace alpaka
{
    struct ConceptAcc
    {
    };

    //! True if TAcc is an accelerator, i.e. if it implements the ConceptAcc concept.
    template<typename TAcc>
    inline constexpr bool isAccelerator = concepts::ImplementsConcept<ConceptAcc, TAcc>::value;

    //! The accelerator traits.
    namespace trait
    {
        //! The accelerator type trait.
        template<typename T, typename TSfinae = void>
        struct AccType;

        //! The single thread accelerator trait.
        //!
        //! If TAcc is an accelerator that supports only a single thread per block, inherit from std::true_type.
        //! If TAcc is not an accelerator, or an accelerator that supports multiple threads per block, inherit from
        //! std::false_type.
        template<typename TAcc, typename TSfinae = void>
        struct IsSingleThreadAcc : std::false_type
        {
        };

        //! The multi thread accelerator trait.
        //!
        //! If TAcc is an accelerator that supports multiple threads per block, inherit from std::true_type.
        //! If TAcc is not an accelerator, or an accelerator that supports only a single thread per block, inherit from
        //! std::false_type.
        template<typename TAcc, typename TSfinae = void>
        struct IsMultiThreadAcc : std::false_type
        {
        };

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
                return core::demangled<TAcc>;
            }
        };
    } // namespace trait

    //! The accelerator type trait alias template to remove the ::type.
    template<typename T>
    using Acc = typename trait::AccType<T>::type;

    //! True if TAcc is an accelerator that supports only a single thread per block, false otherwise.
    template<typename TAcc>
    inline constexpr bool isSingleThreadAcc = trait::IsSingleThreadAcc<TAcc>::value;

    //! True if TAcc is an accelerator that supports multiple threads per block, false otherwise.
    template<typename TAcc>
    inline constexpr bool isMultiThreadAcc = trait::IsMultiThreadAcc<TAcc>::value;

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

    namespace trait
    {
        template<typename TAcc, typename TProperty>
        struct QueueType<TAcc, TProperty, std::enable_if_t<concepts::ImplementsConcept<ConceptAcc, TAcc>::value>>
        {
            using type = typename QueueType<typename alpaka::trait::PlatformType<TAcc>::type, TProperty>::type;
        };

    } // namespace trait

} // namespace alpaka
