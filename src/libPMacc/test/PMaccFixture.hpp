/**
 * Copyright 2016-2016 Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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

#pragma once

#include <Environment.hpp>
#include <dimensions/DataSpace.hpp>

/** Fixture that initializes libPMacc for a given dimensionality */
template<unsigned T_dim>
struct PMaccFixture
{
    PMaccFixture()
    {
        const PMacc::DataSpace<T_dim> devices = PMacc::DataSpace<T_dim>::create(1);
        const PMacc::DataSpace<T_dim> periodic = PMacc::DataSpace<T_dim>::create(1);
        PMacc::Environment<T_dim>::get().initDevices(devices, periodic);
    }
};

typedef PMaccFixture<2> PMaccFixture2D;
typedef PMaccFixture<3> PMaccFixture3D;
