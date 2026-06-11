/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/filter/generic/Free.def"
#include "picongpu/particles/functor/User.hpp"

#include <string>

namespace picongpu
{
    namespace particles
    {
        namespace filter
        {
            namespace generic
            {
                namespace acc
                {
                    /** wrapper for the user filter on the accelerator
                     *
                     * @tparam T_Functor user defined filter
                     */
                    template<typename T_Functor>
                    struct Free : private T_Functor
                    {
                        //! type of the user filter
                        using Functor = T_Functor;

                        //! store user filter instance
                        HDINLINE Free(Functor const& filter) : Functor(filter)
                        {
                        }

                        /** execute the user filter
                         *
                         * @tparam T_Args type of the arguments passed to the user filter
                         *
                         * @param particle particle to use for the filtering
                         */
                        template<typename T_Worker, typename T_Particle>
                        HDINLINE bool operator()(T_Worker const&, T_Particle const& particle)
                        {
                            bool const isValid = particle.isHandleValid();

                            return isValid && Functor::operator()(particle);
                        }
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

                    /** create device filter
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

                    HINLINE static std::string getName()
                    {
                        // provide the name from the user functor
                        return Functor::name;
                    }

                    /** A filter is deterministic if the filter outcome is equal between evaluations. If so, set this
                     * variable to true, otherwise to false.
                     *
                     * Example: A filter were results depend on a random number generator must return false.
                     */
                    static constexpr bool isDeterministic = Functor::isDeterministic;
                };

            } // namespace generic
        } // namespace filter
    } // namespace particles
} // namespace picongpu
