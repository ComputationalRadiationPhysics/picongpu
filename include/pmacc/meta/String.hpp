/* Copyright 2018-2021 Rene Widera
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

#include <boost/preprocessor/repetition/repeat_from_to.hpp>


namespace pmacc
{
    namespace meta
    {
        /** get character of an C-string
         *
         * @tparam T_len length of the string
         *
         * @param cstr input string
         * @param idx index of the character
         * @return if x < T_len character at index idx, else '0'
         */
        template<int T_len>
        constexpr auto elem_at(char const (&cstr)[T_len], size_t const idx) -> char
        {
            return idx < T_len ? cstr[idx] : 0;
        }

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


#define PMACC_CHAR_AT_N(z, n, name) pmacc::meta::elem_at<sizeof(name)>(name, n),

        /** create a compile time string type
         *
         * Support strings with up to 64 characters.
         * Longer strings are cropped to 64 characters.
         *
         * usage example:
         * @code{.cpp}
         * // create an instance of the compile time string
         * auto particleName = PMACC_CSTRING( "electrons" ){};
         * // create a C++ type (can be used as template parameter)
         * using Electrons = PMACC_CSTRING( "electrons" );
         * @endcode
         */

#define PMACC_CSTRING(str)                                                                                            \
    /* // PMACC_CSTRING("example") is transformed in                                                                  \
     * pmacc::meta::String<                                                                                           \
     *     pmacc::meta::elem_at< sizeof("example") >( sizeof("example", 0 ),                                          \
     *     pmacc::meta::elem_at< sizeof("example") >( sizeof("example", 1 ),                                          \
     *     ...                                                                                                        \
     *     pmacc::meta::elem_at< sizeof("example") >( sizeof("example", 63 ),                                         \
     *     0                                                                                                          \
     * >                                                                                                              \
     */                                                                                                               \
    pmacc::meta::String<BOOST_PP_REPEAT_FROM_TO(                                                                      \
        0, /* support up to 64 charactres */                                                                          \
        64,                                                                                                           \
        PMACC_CHAR_AT_N,                                                                                              \
        str) /* add a end zero because PMACC_CHAR_AT_N end with a comma */                                            \
                        0>

    } // namespace meta
} // namespace pmacc
