/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <type_traits>

namespace alpaka
{
    namespace concepts
    {
        //#############################################################################
        //! Tag used in class inheritance hierarchies that describes that a specific concept (TConcept)
        //! is implemented by the given base class (TBase).
        template<
            typename TConcept,
            typename TBase>
        struct Implements
        {
        };

        //#############################################################################
        //! Checks whether the concept is implemented by the given class
        template<
            typename TConcept,
            typename TDerived>
        struct ImplementsConcept {
            template<
                typename TBase>
            static auto implements(Implements<TConcept, TBase>&) -> std::true_type;
            static auto implements(...) -> std::false_type;

            static constexpr auto value = decltype(implements(std::declval<TDerived&>()))::value;
        };

        namespace detail
        {
            //#############################################################################
            //! Returns the type that implements the given concept in the inheritance hierarchy.
            template<
                typename TConcept,
                typename TDerived,
                typename Sfinae = void>
            struct ImplementationBaseType;

            //#############################################################################
            //! Base case for types that do not inherit from "Implements<TConcept, ...>" is the type itself.
            template<
                typename TConcept,
                typename TDerived>
            struct ImplementationBaseType<
                TConcept,
                TDerived,
                typename std::enable_if<!ImplementsConcept<TConcept, TDerived>::value>::type>
            {
                using type = TDerived;
            };

            //#############################################################################
            //! For types that inherit from "Implements<TConcept, ...>" it finds the base class (TBase) which implements the concept.
            template<
                typename TConcept,
                typename TDerived>
            struct ImplementationBaseType<
                TConcept,
                TDerived,
                typename std::enable_if<ImplementsConcept<TConcept, TDerived>::value>::type>
            {
                template<
                    typename TBase>
                static auto implementer(Implements<TConcept, TBase>&) -> TBase;

                using type = decltype(implementer(std::declval<TDerived&>()));

                static_assert(std::is_base_of<type, TDerived>::value, "The type implementing the concept has to be a publicly accessible base class!");
            };
        }

        //#############################################################################
        //! Returns the type that implements the given concept in the inheritance hierarchy.
        template<
            typename TConcept,
            typename TDerived>
        using ImplementationBase = typename detail::ImplementationBaseType<TConcept, TDerived>::type;
    }
}
