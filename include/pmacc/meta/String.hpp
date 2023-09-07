/* Copyright 2018-2023 Rene Widera
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

#include <boost/metaparse/string.hpp>

#include <array>
#include <string>

namespace pmacc
{
    namespace meta
    {
        /** compile time string
         *
         * The size of the instance is 1 byte.
         */
        template<char... T_c>
        struct String
        {
            /** get stored string */
            static auto str() -> std::string
            {
                return std::string(std::array<char, sizeof...(T_c) + 1>({T_c...,
                                                                         // at terminal zero to support empty strings
                                                                         0})
                                       .data());
            }
        };

        namespace internal
        {
            template<typename T>
            struct MakeString;

            template<char... T_c>
            struct MakeString<boost::metaparse::string<T_c...>>
            {
                using type = String<T_c...>;
            };
        } // namespace internal

        /** create a compile time string type
         *
         * usage example:
         * @code{.cpp}
         * // create an instance of the compile time string
         * auto particleName = PMACC_CSTRING( "electrons" ){};
         * // create a C++ type (can be used as template parameter)
         * using Electrons = PMACC_CSTRING( "electrons" );
         * @endcode
         */
#define PMACC_CSTRING(str) typename ::pmacc::meta::internal::MakeString<BOOST_METAPARSE_STRING(str)>::type

    } // namespace meta
} // namespace pmacc
