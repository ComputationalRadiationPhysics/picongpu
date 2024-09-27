/* Copyright 2013-2023 Rene Widera, Axel Huebl
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
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
