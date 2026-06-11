/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/filter/Interface.hpp"
#include "pmacc/lockstep/Worker.hpp"
#include "pmacc/types.hpp"

#include <string>

namespace pmacc
{
    namespace functor
    {
        namespace acc
        {
            /** interface to combine a filter and a functor on the accelerator
             *
             * @tparam T_FilterOperator pmacc::filter::operators, type concatenate the
             *                          results of the filter
             * @tparam T_Filter pmacc::filter::Interface, type of the filter
             * @tparam T_Functor pmacc::functor::Interface, type of the functor
             */
            template<typename T_FilterOperator, typename T_Filter, typename T_Functor>
            struct Filtered
                : private T_Filter
                , public T_Functor
            {
                using Filter = T_Filter;
                using Functor = T_Functor;

                HDINLINE Filtered(Filter const& filter, Functor const& functor) : Filter(filter), Functor(functor)
                {
                }

                /** execute the functor depending of the filter result
                 *
                 * Call the filter for each argument. If the combined result is true
                 * the user functor is called.
                 *
                 * @param args arguments passed to the functor if the filter results of
                 *             each argument evaluate to true when combined
                 */
                template<typename T_Worker, typename... T_Args>
                HDINLINE auto operator()(T_Worker const& worker, T_Args&&... args) -> void
                {
                    // call the filter on each argument and combine the results
                    bool const combinedResult
                        = T_FilterOperator{}((*static_cast<Filter*>(this))(worker, std::forward<T_Args>(args))...);

                    if(combinedResult)
                        (*static_cast<Functor*>(this))(worker, args...);
                }
            };

        } // namespace acc

        /** combine a filter and a functor
         *
         * Creates a functor where each argument which is passed to
         * the accelerator instance is evaluated by the filter and if the
         * combined result is true the functor is executed.
         *
         * @tparam T_FilterOperator pmacc::filter::operators, type concatenate the
         *                          results of the filter
         * @tparam T_Filter pmacc::filter::Interface, type of the filter
         * @tparam T_Functor pmacc::functor::Interface, type of the functor
         */
        template<typename T_FilterOperator, typename T_Filter, typename T_Functor>
        struct Filtered;

        /** specialization of Filtered (with unary filter)
         *
         * This specialization can only be used if T_Filter is of the type pmacc::filter::Interface
         * and T_Functor is of the type pmacc::functor::Interface.
         * A unary filters means that each argument can only pass the same filter
         * check before its results are combined.
         */
        template<
            typename T_FilterOperator,
            typename T_Filter,
            typename T_Functor,
            uint32_t T_numFunctorArguments

            >
        struct Filtered<
            T_FilterOperator,
            filter::Interface<T_Filter, 1u>,
            Interface<T_Functor, T_numFunctorArguments, void>>
            : private filter::Interface<T_Filter, 1u>
            , Interface<T_Functor, T_numFunctorArguments, void>
        {
            template<typename... T_Params>
            struct apply
            {
                using type = Filtered<
                    T_FilterOperator,
                    typename boost::mpl::apply<T_Filter, T_Params...>::type,
                    typename boost::mpl::apply<T_Functor, T_Params...>::type>;
            };

            using Filter = filter::Interface<T_Filter, 1u>;
            using Functor = Interface<T_Functor, T_numFunctorArguments, void>;

            template<typename DeferFunctor = Functor>
            HINLINE Filtered(uint32_t const currentStep, IdGenerator idGen)
                : Filter(currentStep, idGen)
                , Functor(currentStep, idGen)
            {
            }

            /** create a filtered functor which can be used on the accelerator
             *
             * @tparam T_OffsetType type to describe the size of a domain
             * @tparam T_Worker lockstep worker type
             *
             * @param worker lockstep worker
             * @param domainOffset offset to the origin of the local domain
             *                     This can be e.g a supercell or cell offset and depends
             *                     of the context where the interface is specialized.
             * @return accelerator instance of the filtered functor
             */
            template<typename T_OffsetType, typename T_Worker>
            HDINLINE auto operator()(T_Worker const& worker, T_OffsetType const& domainOffset) const -> acc::Filtered<
                T_FilterOperator,
                decltype(alpaka::core::declval<Filter>()(worker, domainOffset)),
                decltype(alpaka::core::declval<Functor>()(worker, domainOffset))>
            {
                return acc::Filtered<
                    T_FilterOperator,
                    decltype(alpaka::core::declval<Filter>()(worker, domainOffset)),
                    decltype(alpaka::core::declval<Functor>()(worker, domainOffset))>(
                    (*static_cast<Filter const*>(this))(worker, domainOffset),
                    (*static_cast<Functor const*>(this))(worker, domainOffset));
            }

            /** get name the of the filtered functor
             *
             * @return combination of the filter and functor name, the names are
             *         separated by an underscore `_`
             */
            HINLINE std::string getName() const
            {
                return Filter::getName() + std::string("_") + Functor::getName();
            }
        };

    } // namespace functor
} // namespace pmacc
