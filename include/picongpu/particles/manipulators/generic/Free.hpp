/*
 * SPDX-FileCopyrightText: Rene Widera, Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/functor/User.hpp"
#include "picongpu/particles/manipulators/generic/Free.def"

#include <type_traits>
#include <utility>

namespace picongpu
{
    namespace particles
    {
        namespace manipulators
        {
            namespace generic
            {
                namespace acc
                {
                    /** wrapper for the user manipulator functor on the accelerator
                     *
                     * @tparam T_Functor user defined manipulators
                     */
                    template<typename T_Functor>
                    struct Free : private T_Functor
                    {
                        //! type of the user manipulators
                        using Functor = T_Functor;

                        //! store user manipulators instance
                        HDINLINE Free(Functor const& manipulators) : Functor(manipulators)
                        {
                        }

                        /** execute the user manipulator functor
                         *
                         * @tparam T_Worker lockstep worker type
                         *
                         * @param args arguments passed to the user functor
                         *
                         * @{
                         */
                        template<typename T_Worker, typename... T_Args>
                        HDINLINE auto operator()(T_Worker const& worker, T_Args&&... args)
                            -> decltype(std::declval<Functor>()(worker, std::forward<T_Args>(args)...))
                        {
                            Functor::operator()(worker, std::forward<T_Args>(args)...);
                        }

                        template<typename T_Worker, typename... T_Args>
                        HDINLINE auto operator()(T_Worker const&, T_Args&&... args)
                            -> decltype(std::declval<Functor>()(std::forward<T_Args>(args)...))
                        {
                            Functor::operator()(std::forward<T_Args>(args)...);
                        }

                        /** @} */
                    };
                } // namespace acc

                template<typename T_Functor>
                struct Free : protected functor::User<T_Functor>
                {
                    using Functor = functor::User<T_Functor>;

                    template<typename T_SpeciesType>
                    struct apply
                    {
                        using type = Free;
                    };

                    /** constructor
                     *
                     * @param currentStep current simulation time step
                     */
                    HINLINE Free(uint32_t currentStep, IdGenerator idGen) : Functor(currentStep, idGen)
                    {
                    }

                    /** create device manipulator functor
                     *
                     * @tparam T_Worker lockstep worker type
                     *
                     * @param worker lockstep worker
                     * @param offset (in supercells, without any guards) to the
                     *         origin of the local domain
                     * @param configuration of the worker
                     */
                    template<typename T_Worker>
                    HDINLINE acc::Free<Functor> operator()(T_Worker const&, DataSpace<simDim> const&) const
                    {
                        return acc::Free<Functor>(*static_cast<Functor const*>(this));
                    }

                    //! get the name of the functor
                    HINLINE static std::string getName()
                    {
                        // we provide the name from the param class
                        return Functor::name;
                    }
                };

            } // namespace generic
        } // namespace manipulators
    } // namespace particles
} // namespace picongpu
