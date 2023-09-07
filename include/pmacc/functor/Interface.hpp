/* Copyright 2017-2023 Rene Widera, Pawel Ordyna
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/lockstep/Worker.hpp"
#include "pmacc/static_assert.hpp"
#include "pmacc/types.hpp"

#include <boost/core/ignore_unused.hpp>

#include <string>
#include <type_traits>


namespace pmacc
{
    namespace functor
    {
        namespace acc
        {
            /** functor interface used on the accelerator side
             *
             * The user functor of the type T_UserFunctor must contain
             * - the `operator()` with T_numArguments arguments and a return type T_ReturnType.
             * - a copy constructor
             * This interface is used to wrap the user functor to make sure that
             * the required interface is fulfilled.
             *
             * @tparam T_UserFunctor user functor type
             * @tparam T_numArguments number which must be supported by T_UserFunctor
             * @tparam T_ReturnType required return type of T_UserFunctor
             */
            template<typename T_UserFunctor, uint32_t T_numArguments, typename T_ReturnType>
            struct Interface : public T_UserFunctor
            {
                //! type of the user functor
                using UserFunctor = T_UserFunctor;

                /** constructor
                 *
                 * @param functor user functor instance
                 */
                HDINLINE Interface(UserFunctor const& functor) : UserFunctor(functor)
                {
                }

                /** execute the functor
                 *
                 * The number of arguments and the return type of the user functor are
                 * evaluated at compile-time and must be equal to the interface description.
                 *
                 * @tparam T_Args type of the arguments passed to the user functor
                 *
                 * @param args arguments passed to the user functor
                 * @return T_ReturnType
                 */
                template<typename T_Worker, typename... T_Args>
                HDINLINE auto operator()(T_Worker const& worker, T_Args&&... args) -> T_ReturnType
                {
                    /* check if the current used number of arguments to execute the
                     * functor is equal to the interface requirements
                     */
                    PMACC_CASSERT_MSG_TYPE(
                        __user_functor_has_wrong_number_of_arguments,
                        UserFunctor,
                        T_numArguments == sizeof...(args));

                    // get the return type of the user functor
                    using UserFunctorReturnType = decltype(alpaka::core::declval<UserFunctor>()(worker, args...));

                    // compare user functor return type with the interface requirements
                    PMACC_CASSERT_MSG(
                        __wrong_user_functor_return_type,
                        std::is_same_v<UserFunctorReturnType, T_ReturnType>);
                    return (*static_cast<UserFunctor*>(this))(worker, args...);
                }
            };

        } // namespace acc

        /** Interface for a user functor
         *
         * @tparam T_UserFunctor user functor type
         * @tparam T_numArguments number of arguments which must be supported by T_UserFunctor
         * @tparam T_ReturnType required return type of T_UserFunctor
         */
        template<typename T_UserFunctor, uint32_t T_numArguments, typename T_ReturnType>
        struct Interface : private T_UserFunctor
        {
            //! type of the user functor
            using UserFunctor = T_UserFunctor;

            /** constructor
             *
             * This constructor is only compiled if the user functor has
             * a host side constructor with one (uint32_t) argument.
             *
             * @tparam DeferFunctor is used to defer the functor type evaluation to enable/disable
             *                      the constructor
             * @param currentStep current simulation time step
             * @param is used to enable/disable the constructor (do not pass any value to this parameter)
             */
            template<typename DeferFunctor = UserFunctor>
            HINLINE Interface(
                uint32_t const currentStep,
                std::enable_if_t<std::is_constructible_v<DeferFunctor, uint32_t>>* = nullptr)
                : UserFunctor(currentStep)
            {
            }

            /** constructor
             *
             * This constructor is only compiled if the user functor has a default constructor.
             *
             * @tparam DeferFunctor is used to defer the functor type evaluation to enable/disable
             *                      the constructor
             * @param currentStep simulation time step
             * @param is used to enable/disable the constructor (do not pass any value to this parameter)
             */
            template<typename DeferFunctor = UserFunctor>
            HINLINE Interface(
                uint32_t const currentStep,
                std::enable_if_t<std::is_constructible_v<DeferFunctor>>* = nullptr)
                : UserFunctor()
            {
                boost::ignore_unused(currentStep);
            }

            /** create a functor which can be used on the accelerator
             *
             * @tparam T_OffsetType type to describe the size of a domain
             * @tparam T_Worker lockstep worker type
             * @tparam T_Args type of the arguments passed to the  functor
             *
             * @param worker lockstep worker
             * @param domainOffset offset to the origin of the local domain
             *                     This can be e.g a supercell or cell offset and depends
             *                     of the context where the interface is specialized.
             * @param args arguments passed to the functor
             * @return an instance of the user functor wrapped by the accelerator
             *         functor interface
             */
            template<typename T_OffsetType, typename T_Worker, typename... T_Args>
            HDINLINE auto operator()(T_Worker const& worker, T_OffsetType const& domainOffset, T_Args... args) const
                -> acc::Interface<
                    decltype(alpaka::core::declval<UserFunctor>()(worker, domainOffset, args...)),
                    T_numArguments,
                    T_ReturnType>
            {
                return (*static_cast<UserFunctor const*>(this))(worker, domainOffset, args...);
            }

            /** get name of the user functor
             *
             * @return name to identify the functor
             */
            HINLINE static std::string getName()
            {
                return UserFunctor::getName();
            }
        };

    } // namespace functor
} // namespace pmacc
