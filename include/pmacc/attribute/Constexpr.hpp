/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz,
 *                     Alexander Grund
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


/**
 * Visual Studio has a bug with constexpr variables being captured in lambdas as
 * non-constexpr variables, causing build errors. The issue has been verified
 * for versions 14.0 and 15.5 (latest at the moment) and is also reported in
 * https://stackoverflow.com/questions/28763375/using-lambda-captured-constexpr-value-as-an-array-dimension
 * and related issue
 * https://developercommunity.visualstudio.com/content/problem/1997/constexpr-not-implicitely-captured-in-lambdas.html
 *
 * As a workaround (until this is fixed in VS) add a new PMACC_CONSTEXPR_CAPTURE
 * macro for declaring constexpr variables that are captured in lambdas and have
 * to remain constexpr inside a lambda e.g., used as a template argument. Such
 * variables have to be declared with PMACC_CONSTEXPR_CAPTURE instead of
 * constexpr. The macro will be replaced with just constexpr for other compilers
 * and for Visual Studio with static constexpr, which makes it capture properly.
 *
 * Note that this macro is to be used only in very few cases, where not only a
 * constexpr is captured, but also it has to remain constexpr inside a lambda.
 */
#ifdef _MSC_VER
#    define PMACC_CONSTEXPR_CAPTURE static constexpr
#elif(defined __GNUC__) && (__GNUC__ > 7)
// workaround for GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=91377
#    define PMACC_CONSTEXPR_CAPTURE static constexpr
#else
#    define PMACC_CONSTEXPR_CAPTURE constexpr
#endif
