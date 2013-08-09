/**
 * Copyright 2013 Axel Huebl, René Widera
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
 * File:   ppFunctions.hpp
 * Author: widera
 *
 * Created on 13. August 2012, 13:32
 */

#ifndef PPFUNCTIONS_HPP
#define	PPFUNCTIONS_HPP


#define PMACC_MIN(x,y) (((x)<=(y))?x:y)
#define PMACC_MAX(x,y) (((x)>(y))?x:y)


#define PMACC_JOIN_DO(x,y) x##y
#define PMACC_JOIN(x,y) PMACC_JOIN_DO(x,y)

#define PMACC_MAX_DO(what,x,y) (((x)>(y))?x what:y what)
#define PMACC_MIN_DO(what,x,y) (((x)<(y))?x what:y what)

#endif	/* PPFUNCTIONS_HPP */

