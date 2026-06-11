/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/particles/filter/IUnary.def"

#include <pmacc/particles/IdProvider.hpp>

#include <string>

namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            /** execute an unary functor if the name is equal
             *
             * @tparam T_Filter filter class (required interface: `getName( )` and default constructor)
             */
            template<typename T_Filter>
            struct ExecuteIfNameIsEqual
            {
                /** evaluate if functor must executed
                 *
                 * @param filterName name of the filter which should started
                 * @param unaryFunctor any unary functor
                 */
                template<typename T_Kernel, typename... T_Args>
                void operator()(
                    std::string const& filterName,
                    uint32_t const currentStep,
                    pmacc::IdGenerator idGen,
                    T_Kernel const unaryFunctor) const
                {
                    if(filterName == T_Filter::getName())
                        unaryFunctor(particles::filter::IUnary<T_Filter>{currentStep, idGen});
                }
            };
        } // namespace misc
    } // namespace plugins
} // namespace picongpu
