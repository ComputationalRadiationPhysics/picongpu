/**
* \file
* Copyright 2014-2016 Benjamin Worpitz, Erik Zenker
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

//#############################################################################
// Include the whole library.
//#############################################################################

//-----------------------------------------------------------------------------
// version number
#include <alpaka/version.hpp>
//-----------------------------------------------------------------------------
// acc
#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/acc/AccCpuThreads.hpp>
#include <alpaka/acc/AccCpuFibers.hpp>
#include <alpaka/acc/AccCpuTbbBlocks.hpp>
#include <alpaka/acc/AccCpuOmp2Blocks.hpp>
#include <alpaka/acc/AccCpuOmp2Threads.hpp>
#include <alpaka/acc/AccCpuOmp4.hpp>
#include <alpaka/acc/AccGpuCudaRt.hpp>
#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/acc/Traits.hpp>
//-----------------------------------------------------------------------------
// atomic
#include <alpaka/atomic/AtomicCudaBuiltIn.hpp>
#include <alpaka/atomic/AtomicNoOp.hpp>
#include <alpaka/atomic/AtomicOmpCritSec.hpp>
#include <alpaka/atomic/AtomicStlLock.hpp>
#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>
//-----------------------------------------------------------------------------
// block
    //-----------------------------------------------------------------------------
    // shared
        //-----------------------------------------------------------------------------
        // dynamic
        #include <alpaka/block/shared/dyn/BlockSharedMemDynBoostAlignedAlloc.hpp>
        #include <alpaka/block/shared/dyn/BlockSharedMemDynCudaBuiltIn.hpp>
        #include <alpaka/block/shared/dyn/Traits.hpp>
        //-----------------------------------------------------------------------------
        // static
        #include <alpaka/block/shared/st/BlockSharedMemStCudaBuiltIn.hpp>
        #include <alpaka/block/shared/st/BlockSharedMemStMasterSync.hpp>
        #include <alpaka/block/shared/st/BlockSharedMemStNoSync.hpp>
        #include <alpaka/block/shared/st/Traits.hpp>
    //-----------------------------------------------------------------------------
    // sync
    #include <alpaka/block/sync/BlockSyncBarrierFiber.hpp>
    #include <alpaka/block/sync/BlockSyncBarrierOmp.hpp>
    #include <alpaka/block/sync/BlockSyncBarrierThread.hpp>
    #include <alpaka/block/sync/BlockSyncCudaBuiltIn.hpp>
    #include <alpaka/block/sync/BlockSyncNoOp.hpp>
    #include <alpaka/block/sync/Traits.hpp>
//-----------------------------------------------------------------------------
// core
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Align.hpp>
#include <alpaka/core/BarrierThread.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/ConcurrentExecPool.hpp>
#include <alpaka/core/Cuda.hpp>
#include <alpaka/core/Debug.hpp>
#include <alpaka/core/Fibers.hpp>
#include <alpaka/core/OpenMp.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unroll.hpp>
#include <alpaka/core/Utility.hpp>
#include <alpaka/core/Vectorize.hpp>
//-----------------------------------------------------------------------------
// dev
#include <alpaka/dev/DevCudaRt.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/cpu/Wait.hpp>
#include <alpaka/dev/Traits.hpp>
//-----------------------------------------------------------------------------
// dim
#include <alpaka/dim/DimArithmetic.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/dim/Traits.hpp>
//-----------------------------------------------------------------------------
// event
#include <alpaka/event/EventCudaRt.hpp>
#include <alpaka/event/EventCpu.hpp>
#include <alpaka/event/Traits.hpp>
//-----------------------------------------------------------------------------
// exec
#include <alpaka/exec/ExecCpuSerial.hpp>
#include <alpaka/exec/ExecCpuThreads.hpp>
#include <alpaka/exec/ExecCpuFibers.hpp>
#include <alpaka/exec/ExecCpuTbbBlocks.hpp>
#include <alpaka/exec/ExecCpuOmp2Blocks.hpp>
#include <alpaka/exec/ExecCpuOmp2Threads.hpp>
#include <alpaka/exec/ExecCpuOmp4.hpp>
#include <alpaka/exec/ExecGpuCudaRt.hpp>
#include <alpaka/exec/Traits.hpp>
//-----------------------------------------------------------------------------
// extent
#include <alpaka/extent/Traits.hpp>
//-----------------------------------------------------------------------------
// idx
#include <alpaka/idx/bt/IdxBtCudaBuiltIn.hpp>
#include <alpaka/idx/bt/IdxBtOmp.hpp>
#include <alpaka/idx/bt/IdxBtRefFiberIdMap.hpp>
#include <alpaka/idx/bt/IdxBtRefThreadIdMap.hpp>
#include <alpaka/idx/bt/IdxBtZero.hpp>
#include <alpaka/idx/gb/IdxGbCudaBuiltIn.hpp>
#include <alpaka/idx/gb/IdxGbRef.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/idx/MapIdx.hpp>
//-----------------------------------------------------------------------------
// kernel
#include <alpaka/kernel/Traits.hpp>
//-----------------------------------------------------------------------------
// math
#include <alpaka/math/MathCudaBuiltIn.hpp>
#include <alpaka/math/MathStl.hpp>
//-----------------------------------------------------------------------------
// mem
#include <alpaka/mem/alloc/AllocCpuBoostAligned.hpp>
#include <alpaka/mem/alloc/AllocCpuNew.hpp>
#include <alpaka/mem/alloc/Traits.hpp>

#include <alpaka/mem/buf/BufCudaRt.hpp>
#include <alpaka/mem/buf/BufCpu.hpp>
#include <alpaka/mem/buf/Traits.hpp>

#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/mem/view/ViewStdContainers.hpp>
#include <alpaka/mem/view/ViewSubView.hpp>
#include <alpaka/mem/view/Traits.hpp>
//-----------------------------------------------------------------------------
// meta
#include <alpaka/meta/Apply.hpp>
#include <alpaka/meta/ApplyTuple.hpp>
#include <alpaka/meta/CartesianProduct.hpp>
#include <alpaka/meta/Concatenate.hpp>
#include <alpaka/meta/DependentFalseType.hpp>
#include <alpaka/meta/Filter.hpp>
#include <alpaka/meta/Fold.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/meta/IntegerSequence.hpp>
#include <alpaka/meta/IsIntegralSuperset.hpp>
#include <alpaka/meta/IsStrictBase.hpp>
#include <alpaka/meta/Metafunctions.hpp>
#include <alpaka/meta/NdLoop.hpp>
#include <alpaka/meta/Set.hpp>
#include <alpaka/meta/StdTupleAsMplSequence.hpp>
#include <alpaka/meta/Transform.hpp>
//-----------------------------------------------------------------------------
// offset
#include <alpaka/offset/Traits.hpp>
//-----------------------------------------------------------------------------
// platform
#include <alpaka/pltf/PltfCpu.hpp>
#include <alpaka/pltf/PltfCudaRt.hpp>
#include <alpaka/pltf/Traits.hpp>
//-----------------------------------------------------------------------------
// rand
#include <alpaka/rand/RandCuRand.hpp>
#include <alpaka/rand/RandStl.hpp>
#include <alpaka/rand/Traits.hpp>
//-----------------------------------------------------------------------------
// size
#include <alpaka/size/Traits.hpp>
//-----------------------------------------------------------------------------
// stream
#include <alpaka/stream/StreamCudaRtAsync.hpp>
#include <alpaka/stream/StreamCudaRtSync.hpp>
#include <alpaka/stream/StreamCpuAsync.hpp>
#include <alpaka/stream/StreamCpuSync.hpp>
#include <alpaka/stream/Traits.hpp>
//-----------------------------------------------------------------------------
// time
#include <alpaka/time/Traits.hpp>
//-----------------------------------------------------------------------------
// wait
#include <alpaka/wait/Traits.hpp>
//-----------------------------------------------------------------------------
// workdiv
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <alpaka/workdiv/Traits.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>
//-----------------------------------------------------------------------------
// vec
#include <alpaka/vec/Vec.hpp>
#include <alpaka/vec/Traits.hpp>
