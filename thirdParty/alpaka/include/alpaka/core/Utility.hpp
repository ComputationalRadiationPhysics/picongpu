/**
* \file
* Copyright 2017 Rene Widera
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Common.hpp>

#if BOOST_LANG_CUDA && BOOST_COMP_CLANG_CUDA
#   include <type_traits> // std::add_rvalue_reference
#else
#   include <utility> // std::declval
#endif


namespace alpaka
{
    namespace core
    {
        //-----------------------------------------------------------------------------
        //! convert any type to a reverence type
        //
        // This function is equivalent to std::declval() but can be used
        // within an alpaka accelerator kernel too.
        // This function can be used only within std::decltype().
        //-----------------------------------------------------------------------------
#if BOOST_LANG_CUDA && BOOST_COMP_CLANG_CUDA
        template< class T >
        ALPAKA_FN_HOST_ACC
        typename std::add_rvalue_reference<T>::type
        declval();
#else
        using std::declval;
#endif
    }
}
