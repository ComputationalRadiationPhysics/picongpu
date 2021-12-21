/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace alpaka
{
    //#############################################################################
    //! Defines the parallelism hierarchy levels of alpaka
    namespace hierarchy
    {
        struct Grids
        {
        };

        struct Blocks
        {
        };

        struct Threads
        {
        };
    } // namespace hierarchy
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
    } // namespace origin
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
    } // namespace unit

    using namespace origin;
    using namespace unit;
} // namespace alpaka
