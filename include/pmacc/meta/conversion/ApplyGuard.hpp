/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

namespace pmacc
{
    /** prevent a type for boost apply
     *
     * Avoid that boost apply is called for a given type.
     *
     * @tparam T_LockedType type where a boost::apply should not be performed
     *
     * @code{.cpp}
     *
     * struct Foo{};
     * struct Bar{};
     *
     * using Result = boost::apply1<
     *     ApplyGuard< Foo >,
     *     Bar
     * >;
     * PMACC_CASSERT(
     *  boost::is_same<
     *      Foo,
     *      Result
     *  >::value
     * );
     * @endcode
     */
    template<typename T_LockedType>
    struct ApplyGuard
    {
        template<typename... T_Args>
        struct apply
        {
            using type = T_LockedType;
        };
    };

} // namespace pmacc
