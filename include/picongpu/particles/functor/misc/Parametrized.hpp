/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace functor
        {
            namespace misc
            {
                template<typename T_Parameters>
                struct Parametrized
                {
                    //! Parameters type
                    using Parameters = T_Parameters;

                    //! Construct a functor, copy static parameters() into the member
                    HINLINE Parametrized() : m_parameters(parameters())
                    {
                    }

                    //! Pass the parameters from the host side by changing this value
                    static Parameters& parameters()
                    {
                        static auto staticParameters = Parameters{};
                        return staticParameters;
                    }

                protected:
                    //! Parameters values to be accessed on the device side
                    T_Parameters m_parameters;
                };

            } // namespace misc
        } // namespace functor
    } // namespace particles
} // namespace picongpu
