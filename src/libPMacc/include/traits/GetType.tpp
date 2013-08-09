/**
 * Copyright 2013 René Widera
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
 
/* 
 * File:   GetType.tpp
 * Author: widera
 *
 * Created on 4. Februar 2013, 10:06
 */


#pragma once


namespace PMacc
{
namespace traits
{

template<>
struct GetType<float>
{
    typedef float type;
};

template<>
struct GetType<double>
{
    typedef double type;
};

template<>
struct GetType<int>
{
    typedef int type;
};


}//namespace traits

}//namepsace PMacc

