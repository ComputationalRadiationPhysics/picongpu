/**
* \file
* Copyright 2014-2016 Benjamin Worpitz, Rene Widera
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

namespace alpaka
{
    //#############################################################################
    //! Defines the parallelism hierarchy levels of Alpaka
    namespace hierarchy
    {
        struct Grids{};

        struct Blocks{};

        struct Threads{};
    }
    //-----------------------------------------------------------------------------
    //! Defines the origins available for getting extent and indices of kernel executions.
    namespace origin
    {
        //#############################################################################
        //! This type is used to get the extents/indices relative to the grid.
        struct Grid;
        //#############################################################################
        //! This type is used to get the extent/indices relative to a/the current block.
        struct Block;
        //#############################################################################
        //! This type is used to get the extents relative to the thread.
        struct Thread;
    }
    //-----------------------------------------------------------------------------
    //! Defines the units available for getting extent and indices of kernel executions.
    namespace unit
    {
        //#############################################################################
        //! This type is used to get the extent/indices in units of blocks.
        struct Blocks;
        //#############################################################################
        //! This type is used to get the extent/indices in units of threads.
        struct Threads;
        //#############################################################################
        //! This type is used to get the extents/indices in units of elements.
        struct Elems;
    }

    using namespace origin;
    using namespace unit;
}
