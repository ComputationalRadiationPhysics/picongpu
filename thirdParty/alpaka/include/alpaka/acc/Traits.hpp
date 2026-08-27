/* Copyright 2024 Benjamin Worpitz, Bernhard Manfred Gruber, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccDevProps.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/DemangleTypeNames.hpp"
#include "alpaka/core/Interface.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/platform/Traits.hpp"
#include "alpaka/queue/Traits.hpp"

#include <string>
#include <type_traits>
#include <typeinfo>

namespace alpaka
{
    struct InterfaceAcc
    {
    };

    namespace concepts
    {
        template<typename T>
        concept Acc = requires { requires alpaka::interface::ImplementsInterface<alpaka::InterfaceAcc, T>::value; };
    } // namespace concepts

    //! True if TAcc is an accelerator, i.e. if it implements the InterfaceAcc concept.
    template<typename TAcc>
    [[deprecated("use the alpaka::concepts::Acc instead.")]] inline constexpr bool isAccelerator = concepts::Acc<TAcc>;

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
        template<concepts::Acc TAcc>
        struct GetAccDevProps;

        //! The accelerator name trait.
        //!
        //! The default implementation returns the mangled class name.
        template<concepts::Acc TAcc>
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
    template<concepts::Acc TAcc, typename TDev>
    ALPAKA_FN_HOST auto getAccDevProps(TDev const& dev) -> AccDevProps<Dim<TAcc>, Idx<TAcc>>
    {
        using ImplementationBase = interface::ImplementationBase<InterfaceAcc, TAcc>;
        return trait::GetAccDevProps<ImplementationBase>::getAccDevProps(dev);
    }

    //! \return The accelerator name
    //!
    //! \tparam TAcc The accelerator type.
    template<concepts::Acc TAcc>
    ALPAKA_FN_HOST auto getAccName() -> std::string
    {
        return trait::GetAccName<TAcc>::getAccName();
    }

    namespace trait
    {
        template<concepts::Acc TAcc, typename TProperty>
        struct QueueType<TAcc, TProperty>
        {
            using type = typename QueueType<typename alpaka::trait::PlatformType<TAcc>::type, TProperty>::type;
        };

    } // namespace trait

} // namespace alpaka
