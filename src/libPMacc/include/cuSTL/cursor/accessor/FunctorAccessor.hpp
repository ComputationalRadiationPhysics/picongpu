/**
 * Copyright 2013 Heiko Burau, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * libPMacc is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License and the GNU Lesser General Public License 
 * for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * and the GNU Lesser General Public License along with libPMacc. 
 * If not, see <http://www.gnu.org/licenses/>. 
 */ 
 
#ifndef CURSOR_FUNCTORACCESSOR_HPP
#define CURSOR_FUNCTORACCESSOR_HPP

#include <boost/type_traits/add_const.hpp>
#include <forward.hpp>
#include "types.h"

namespace PMacc
{
namespace cursor
{
    
template<typename _Functor, typename ArgType>
struct FunctorAccessor
{
    _Functor functor;
    
    typedef typename ::PMacc::result_of::Functor<_Functor, ArgType>::type type;
    
    HDINLINE FunctorAccessor(const _Functor& functor) : functor(functor) {}
    
    template<typename TCursor>
    HDINLINE type operator()(TCursor& cursor)
    {
        return this->functor(forward(*cursor));
    }
};
    
} // cursor
} // PMacc

#endif // CURSOR_FUNCTORACCESSOR_HPP
