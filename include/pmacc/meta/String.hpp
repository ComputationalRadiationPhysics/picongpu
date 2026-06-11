/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
                return std::string(
                    std::array<char, sizeof...(T_c) + 1>({T_c...,
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
