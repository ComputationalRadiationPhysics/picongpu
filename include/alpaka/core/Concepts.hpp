/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace alpaka::concepts
{
    //! Tag used in class inheritance hierarchies that describes that a specific concept (TConcept)
    //! is implemented by the given base class (TBase).
    template<typename TConcept, typename TBase>
    struct Implements
    {
    };

    //! Checks whether the concept is implemented by the given class
    template<typename TConcept, typename TDerived>
    struct ImplementsConcept
    {
        template<typename TBase>
        static auto implements(Implements<TConcept, TBase>&) -> std::true_type;
        static auto implements(...) -> std::false_type;

        static constexpr auto value = decltype(implements(std::declval<TDerived&>()))::value;
    };

    namespace detail
    {
        //! Returns the type that implements the given concept in the inheritance hierarchy.
        template<typename TConcept, typename TDerived, typename Sfinae = void>
        struct ImplementationBaseType;

        //! Base case for types that do not inherit from "Implements<TConcept, ...>" is the type itself.
        template<typename TConcept, typename TDerived>
        struct ImplementationBaseType<
            TConcept,
            TDerived,
            std::enable_if_t<!ImplementsConcept<TConcept, TDerived>::value>>
        {
            using type = TDerived;
        };

        //! For types that inherit from "Implements<TConcept, ...>" it finds the base class (TBase) which
        //! implements the concept.
        template<typename TConcept, typename TDerived>
        struct ImplementationBaseType<
            TConcept,
            TDerived,
            std::enable_if_t<ImplementsConcept<TConcept, TDerived>::value>>
        {
            template<typename TBase>
            static auto implementer(Implements<TConcept, TBase>&) -> TBase;

            using type = decltype(implementer(std::declval<TDerived&>()));

            static_assert(
                std::is_base_of_v<type, TDerived>,
                "The type implementing the concept has to be a publicly accessible base class!");
        };
    } // namespace detail

    //! Returns the type that implements the given concept in the inheritance hierarchy.
    template<typename TConcept, typename TDerived>
    using ImplementationBase = typename detail::ImplementationBaseType<TConcept, TDerived>::type;
} // namespace alpaka::concepts
