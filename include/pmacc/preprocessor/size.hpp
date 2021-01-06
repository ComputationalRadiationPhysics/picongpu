/* Copyright 2018-2021 Sergei Bastrakov
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


#include <boost/preprocessor/variadic/size.hpp>


/** macro for counting the number of arguments
 *
 * Is only supported for some compilers,
 * for usage check if PMACC_PP_VARIADIC_SIZE is defined.
 * Implementation is essentially the same as BOOST_PP_VARIADIC_SIZE,
 * but supports up to 120 arguments instead of 64.
 * The implementation uses the "paired, sliding arg list" trick
 * explained in https://codecraft.co/2014/11/25/variadic-macros-tricks/
 */
#if(BOOST_PP_VARIADICS == 1)
#    define PMACC_PP_VARIADIC_SIZE_I(                                                                                 \
        e0,                                                                                                           \
        e1,                                                                                                           \
        e2,                                                                                                           \
        e3,                                                                                                           \
        e4,                                                                                                           \
        e5,                                                                                                           \
        e6,                                                                                                           \
        e7,                                                                                                           \
        e8,                                                                                                           \
        e9,                                                                                                           \
        e10,                                                                                                          \
        e11,                                                                                                          \
        e12,                                                                                                          \
        e13,                                                                                                          \
        e14,                                                                                                          \
        e15,                                                                                                          \
        e16,                                                                                                          \
        e17,                                                                                                          \
        e18,                                                                                                          \
        e19,                                                                                                          \
        e20,                                                                                                          \
        e21,                                                                                                          \
        e22,                                                                                                          \
        e23,                                                                                                          \
        e24,                                                                                                          \
        e25,                                                                                                          \
        e26,                                                                                                          \
        e27,                                                                                                          \
        e28,                                                                                                          \
        e29,                                                                                                          \
        e30,                                                                                                          \
        e31,                                                                                                          \
        e32,                                                                                                          \
        e33,                                                                                                          \
        e34,                                                                                                          \
        e35,                                                                                                          \
        e36,                                                                                                          \
        e37,                                                                                                          \
        e38,                                                                                                          \
        e39,                                                                                                          \
        e40,                                                                                                          \
        e41,                                                                                                          \
        e42,                                                                                                          \
        e43,                                                                                                          \
        e44,                                                                                                          \
        e45,                                                                                                          \
        e46,                                                                                                          \
        e47,                                                                                                          \
        e48,                                                                                                          \
        e49,                                                                                                          \
        e50,                                                                                                          \
        e51,                                                                                                          \
        e52,                                                                                                          \
        e53,                                                                                                          \
        e54,                                                                                                          \
        e55,                                                                                                          \
        e56,                                                                                                          \
        e57,                                                                                                          \
        e58,                                                                                                          \
        e59,                                                                                                          \
        e60,                                                                                                          \
        e61,                                                                                                          \
        e62,                                                                                                          \
        e63,                                                                                                          \
        e64,                                                                                                          \
        e65,                                                                                                          \
        e66,                                                                                                          \
        e67,                                                                                                          \
        e68,                                                                                                          \
        e69,                                                                                                          \
        e70,                                                                                                          \
        e71,                                                                                                          \
        e72,                                                                                                          \
        e73,                                                                                                          \
        e74,                                                                                                          \
        e75,                                                                                                          \
        e76,                                                                                                          \
        e77,                                                                                                          \
        e78,                                                                                                          \
        e79,                                                                                                          \
        e80,                                                                                                          \
        e81,                                                                                                          \
        e82,                                                                                                          \
        e83,                                                                                                          \
        e84,                                                                                                          \
        e85,                                                                                                          \
        e86,                                                                                                          \
        e87,                                                                                                          \
        e88,                                                                                                          \
        e89,                                                                                                          \
        e90,                                                                                                          \
        e91,                                                                                                          \
        e92,                                                                                                          \
        e93,                                                                                                          \
        e94,                                                                                                          \
        e95,                                                                                                          \
        e96,                                                                                                          \
        e97,                                                                                                          \
        e98,                                                                                                          \
        e99,                                                                                                          \
        e100,                                                                                                         \
        e101,                                                                                                         \
        e102,                                                                                                         \
        e103,                                                                                                         \
        e104,                                                                                                         \
        e105,                                                                                                         \
        e106,                                                                                                         \
        e107,                                                                                                         \
        e108,                                                                                                         \
        e109,                                                                                                         \
        e110,                                                                                                         \
        e111,                                                                                                         \
        e112,                                                                                                         \
        e113,                                                                                                         \
        e114,                                                                                                         \
        e115,                                                                                                         \
        e116,                                                                                                         \
        e117,                                                                                                         \
        e118,                                                                                                         \
        e119,                                                                                                         \
        size,                                                                                                         \
        ...)                                                                                                          \
        size
#    if BOOST_PP_VARIADICS_MSVC
#        define PMACC_PP_VARIADIC_SIZE(...)                                                                           \
            BOOST_PP_CAT(                                                                                             \
                PMACC_PP_VARIADIC_SIZE_I(                                                                             \
                    __VA_ARGS__,                                                                                      \
                    120,                                                                                              \
                    119,                                                                                              \
                    118,                                                                                              \
                    117,                                                                                              \
                    116,                                                                                              \
                    115,                                                                                              \
                    114,                                                                                              \
                    113,                                                                                              \
                    112,                                                                                              \
                    111,                                                                                              \
                    110,                                                                                              \
                    109,                                                                                              \
                    108,                                                                                              \
                    107,                                                                                              \
                    106,                                                                                              \
                    105,                                                                                              \
                    104,                                                                                              \
                    103,                                                                                              \
                    102,                                                                                              \
                    101,                                                                                              \
                    100,                                                                                              \
                    99,                                                                                               \
                    98,                                                                                               \
                    97,                                                                                               \
                    96,                                                                                               \
                    95,                                                                                               \
                    94,                                                                                               \
                    93,                                                                                               \
                    92,                                                                                               \
                    91,                                                                                               \
                    90,                                                                                               \
                    89,                                                                                               \
                    88,                                                                                               \
                    87,                                                                                               \
                    86,                                                                                               \
                    85,                                                                                               \
                    84,                                                                                               \
                    83,                                                                                               \
                    82,                                                                                               \
                    81,                                                                                               \
                    80,                                                                                               \
                    79,                                                                                               \
                    78,                                                                                               \
                    77,                                                                                               \
                    76,                                                                                               \
                    75,                                                                                               \
                    74,                                                                                               \
                    73,                                                                                               \
                    72,                                                                                               \
                    71,                                                                                               \
                    70,                                                                                               \
                    69,                                                                                               \
                    68,                                                                                               \
                    67,                                                                                               \
                    66,                                                                                               \
                    65,                                                                                               \
                    64,                                                                                               \
                    63,                                                                                               \
                    62,                                                                                               \
                    61,                                                                                               \
                    60,                                                                                               \
                    59,                                                                                               \
                    58,                                                                                               \
                    57,                                                                                               \
                    56,                                                                                               \
                    55,                                                                                               \
                    54,                                                                                               \
                    53,                                                                                               \
                    52,                                                                                               \
                    51,                                                                                               \
                    50,                                                                                               \
                    49,                                                                                               \
                    48,                                                                                               \
                    47,                                                                                               \
                    46,                                                                                               \
                    45,                                                                                               \
                    44,                                                                                               \
                    43,                                                                                               \
                    42,                                                                                               \
                    41,                                                                                               \
                    40,                                                                                               \
                    39,                                                                                               \
                    38,                                                                                               \
                    37,                                                                                               \
                    36,                                                                                               \
                    35,                                                                                               \
                    34,                                                                                               \
                    33,                                                                                               \
                    32,                                                                                               \
                    31,                                                                                               \
                    30,                                                                                               \
                    29,                                                                                               \
                    28,                                                                                               \
                    27,                                                                                               \
                    26,                                                                                               \
                    25,                                                                                               \
                    24,                                                                                               \
                    23,                                                                                               \
                    22,                                                                                               \
                    21,                                                                                               \
                    20,                                                                                               \
                    19,                                                                                               \
                    18,                                                                                               \
                    17,                                                                                               \
                    16,                                                                                               \
                    15,                                                                                               \
                    14,                                                                                               \
                    13,                                                                                               \
                    12,                                                                                               \
                    11,                                                                                               \
                    10,                                                                                               \
                    9,                                                                                                \
                    8,                                                                                                \
                    7,                                                                                                \
                    6,                                                                                                \
                    5,                                                                                                \
                    4,                                                                                                \
                    3,                                                                                                \
                    2,                                                                                                \
                    1, ), )
#    else
#        define PMACC_PP_VARIADIC_SIZE(...)                                                                           \
            PMACC_PP_VARIADIC_SIZE_I(                                                                                 \
                __VA_ARGS__,                                                                                          \
                120,                                                                                                  \
                119,                                                                                                  \
                118,                                                                                                  \
                117,                                                                                                  \
                116,                                                                                                  \
                115,                                                                                                  \
                114,                                                                                                  \
                113,                                                                                                  \
                112,                                                                                                  \
                111,                                                                                                  \
                110,                                                                                                  \
                109,                                                                                                  \
                108,                                                                                                  \
                107,                                                                                                  \
                106,                                                                                                  \
                105,                                                                                                  \
                104,                                                                                                  \
                103,                                                                                                  \
                102,                                                                                                  \
                101,                                                                                                  \
                100,                                                                                                  \
                99,                                                                                                   \
                98,                                                                                                   \
                97,                                                                                                   \
                96,                                                                                                   \
                95,                                                                                                   \
                94,                                                                                                   \
                93,                                                                                                   \
                92,                                                                                                   \
                91,                                                                                                   \
                90,                                                                                                   \
                89,                                                                                                   \
                88,                                                                                                   \
                87,                                                                                                   \
                86,                                                                                                   \
                85,                                                                                                   \
                84,                                                                                                   \
                83,                                                                                                   \
                82,                                                                                                   \
                81,                                                                                                   \
                80,                                                                                                   \
                79,                                                                                                   \
                78,                                                                                                   \
                77,                                                                                                   \
                76,                                                                                                   \
                75,                                                                                                   \
                74,                                                                                                   \
                73,                                                                                                   \
                72,                                                                                                   \
                71,                                                                                                   \
                70,                                                                                                   \
                69,                                                                                                   \
                68,                                                                                                   \
                67,                                                                                                   \
                66,                                                                                                   \
                65,                                                                                                   \
                64,                                                                                                   \
                63,                                                                                                   \
                62,                                                                                                   \
                61,                                                                                                   \
                60,                                                                                                   \
                59,                                                                                                   \
                58,                                                                                                   \
                57,                                                                                                   \
                56,                                                                                                   \
                55,                                                                                                   \
                54,                                                                                                   \
                53,                                                                                                   \
                52,                                                                                                   \
                51,                                                                                                   \
                50,                                                                                                   \
                49,                                                                                                   \
                48,                                                                                                   \
                47,                                                                                                   \
                46,                                                                                                   \
                45,                                                                                                   \
                44,                                                                                                   \
                43,                                                                                                   \
                42,                                                                                                   \
                41,                                                                                                   \
                40,                                                                                                   \
                39,                                                                                                   \
                38,                                                                                                   \
                37,                                                                                                   \
                36,                                                                                                   \
                35,                                                                                                   \
                34,                                                                                                   \
                33,                                                                                                   \
                32,                                                                                                   \
                31,                                                                                                   \
                30,                                                                                                   \
                29,                                                                                                   \
                28,                                                                                                   \
                27,                                                                                                   \
                26,                                                                                                   \
                25,                                                                                                   \
                24,                                                                                                   \
                23,                                                                                                   \
                22,                                                                                                   \
                21,                                                                                                   \
                20,                                                                                                   \
                19,                                                                                                   \
                18,                                                                                                   \
                17,                                                                                                   \
                16,                                                                                                   \
                15,                                                                                                   \
                14,                                                                                                   \
                13,                                                                                                   \
                12,                                                                                                   \
                11,                                                                                                   \
                10,                                                                                                   \
                9,                                                                                                    \
                8,                                                                                                    \
                7,                                                                                                    \
                6,                                                                                                    \
                5,                                                                                                    \
                4,                                                                                                    \
                3,                                                                                                    \
                2,                                                                                                    \
                1, )
#    endif
#endif
