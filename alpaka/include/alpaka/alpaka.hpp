/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Erik Zenker, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
#include <alpaka/acc/AccCpuFibers.hpp>
#include <alpaka/acc/AccCpuOmp2Blocks.hpp>
#include <alpaka/acc/AccCpuOmp2Threads.hpp>
#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/acc/AccCpuTbbBlocks.hpp>
#include <alpaka/acc/AccCpuThreads.hpp>
#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/acc/AccGpuCudaRt.hpp>
#include <alpaka/acc/AccGpuHipRt.hpp>
#include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#include <alpaka/acc/AccOacc.hpp>
#include <alpaka/acc/AccOmp5.hpp>
#include <alpaka/acc/Traits.hpp>
//-----------------------------------------------------------------------------
// atomic
#include <alpaka/atomic/AtomicNoOp.hpp>
#include <alpaka/atomic/AtomicOmpBuiltIn.hpp>
#include <alpaka/atomic/AtomicStdLibLock.hpp>
#include <alpaka/atomic/AtomicUniformCudaHipBuiltIn.hpp>
#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>
//-----------------------------------------------------------------------------
// block
//-----------------------------------------------------------------------------
// shared
//-----------------------------------------------------------------------------
// dynamic
#include <alpaka/block/shared/dyn/BlockSharedMemDynAlignedAlloc.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynMember.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynUniformCudaHipBuiltIn.hpp>
#include <alpaka/block/shared/dyn/Traits.hpp>
//-----------------------------------------------------------------------------
// static
#include <alpaka/block/shared/st/BlockSharedMemStMasterSync.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStMember.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStNoSync.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStUniformCudaHipBuiltIn.hpp>
#include <alpaka/block/shared/st/Traits.hpp>
//-----------------------------------------------------------------------------
// sync
#include <alpaka/block/sync/BlockSyncBarrierFiber.hpp>
#include <alpaka/block/sync/BlockSyncBarrierOmp.hpp>
#include <alpaka/block/sync/BlockSyncBarrierThread.hpp>
#include <alpaka/block/sync/BlockSyncNoOp.hpp>
#include <alpaka/block/sync/BlockSyncUniformCudaHipBuiltIn.hpp>
#include <alpaka/block/sync/Traits.hpp>
//-----------------------------------------------------------------------------
// core
#include <alpaka/core/Align.hpp>
#include <alpaka/core/AlignedAlloc.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/BarrierThread.hpp>
#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/ClipCast.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/ConcurrentExecPool.hpp>
#include <alpaka/core/Cuda.hpp>
#include <alpaka/core/Debug.hpp>
#include <alpaka/core/Fibers.hpp>
#include <alpaka/core/Hip.hpp>
#include <alpaka/core/OmpSchedule.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unroll.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/core/Utility.hpp>
#include <alpaka/core/Vectorize.hpp>
//-----------------------------------------------------------------------------
// dev
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevUniformCudaHipRt.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dev/cpu/Wait.hpp>
//-----------------------------------------------------------------------------
// dim
#include <alpaka/dim/DimArithmetic.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/dim/Traits.hpp>
//-----------------------------------------------------------------------------
// event
#include <alpaka/event/EventCpu.hpp>
#include <alpaka/event/EventOacc.hpp>
#include <alpaka/event/EventOmp5.hpp>
#include <alpaka/event/EventUniformCudaHipRt.hpp>
#include <alpaka/event/Traits.hpp>
//-----------------------------------------------------------------------------
// extent
#include <alpaka/extent/Traits.hpp>
//-----------------------------------------------------------------------------
// idx
#include <alpaka/idx/Accessors.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/idx/bt/IdxBtOmp.hpp>
#include <alpaka/idx/bt/IdxBtRefFiberIdMap.hpp>
#include <alpaka/idx/bt/IdxBtRefThreadIdMap.hpp>
#include <alpaka/idx/bt/IdxBtUniformCudaHipBuiltIn.hpp>
#include <alpaka/idx/bt/IdxBtZero.hpp>
#include <alpaka/idx/gb/IdxGbRef.hpp>
#include <alpaka/idx/gb/IdxGbUniformCudaHipBuiltIn.hpp>
//-----------------------------------------------------------------------------
// kernel
#include <alpaka/kernel/TaskKernelCpuFibers.hpp>
#include <alpaka/kernel/TaskKernelCpuOmp2Blocks.hpp>
#include <alpaka/kernel/TaskKernelCpuOmp2Threads.hpp>
#include <alpaka/kernel/TaskKernelCpuSerial.hpp>
#include <alpaka/kernel/TaskKernelCpuTbbBlocks.hpp>
#include <alpaka/kernel/TaskKernelCpuThreads.hpp>
#include <alpaka/kernel/TaskKernelGpuUniformCudaHipRt.hpp>
#include <alpaka/kernel/TaskKernelOacc.hpp>
#include <alpaka/kernel/TaskKernelOmp5.hpp>
#include <alpaka/kernel/Traits.hpp>
//-----------------------------------------------------------------------------
// math
#include <alpaka/math/MathStdLib.hpp>
#include <alpaka/math/MathUniformCudaHipBuiltIn.hpp>
//-----------------------------------------------------------------------------
// mem
#include <alpaka/mem/alloc/AllocCpuAligned.hpp>
#include <alpaka/mem/alloc/AllocCpuNew.hpp>
#include <alpaka/mem/alloc/Traits.hpp>
#include <alpaka/mem/buf/BufCpu.hpp>
#include <alpaka/mem/buf/BufOacc.hpp>
#include <alpaka/mem/buf/BufOmp5.hpp>
#include <alpaka/mem/buf/BufUniformCudaHipRt.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/view/ViewCompileTimeArray.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/mem/view/ViewStdArray.hpp>
#include <alpaka/mem/view/ViewStdVector.hpp>
#include <alpaka/mem/view/ViewSubView.hpp>
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
#include <alpaka/meta/Functional.hpp>
#include <alpaka/meta/IntegerSequence.hpp>
#include <alpaka/meta/Integral.hpp>
#include <alpaka/meta/IsStrictBase.hpp>
#include <alpaka/meta/Metafunctions.hpp>
#include <alpaka/meta/NdLoop.hpp>
#include <alpaka/meta/Set.hpp>
#include <alpaka/meta/Transform.hpp>
#include <alpaka/meta/Void.hpp>
//-----------------------------------------------------------------------------
// offset
#include <alpaka/offset/Traits.hpp>
//-----------------------------------------------------------------------------
// platform
#include <alpaka/pltf/PltfCpu.hpp>
#include <alpaka/pltf/PltfOacc.hpp>
#include <alpaka/pltf/PltfOmp5.hpp>
#include <alpaka/pltf/PltfUniformCudaHipRt.hpp>
#include <alpaka/pltf/Traits.hpp>
//-----------------------------------------------------------------------------
// rand
#include <alpaka/rand/RandUniformCudaHipRand.hpp>
#include <alpaka/rand/Traits.hpp>
//-----------------------------------------------------------------------------
// idx
#include <alpaka/idx/Traits.hpp>
//-----------------------------------------------------------------------------
// queue
#include <alpaka/queue/Properties.hpp>
#include <alpaka/queue/QueueCpuBlocking.hpp>
#include <alpaka/queue/QueueCpuNonBlocking.hpp>
#include <alpaka/queue/QueueOaccBlocking.hpp>
#include <alpaka/queue/QueueOmp5Blocking.hpp>
#include <alpaka/queue/QueueUniformCudaHipRtBlocking.hpp>
#include <alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp>
#include <alpaka/queue/Traits.hpp>
//-----------------------------------------------------------------------------
// time
#include <alpaka/time/Traits.hpp>
//-----------------------------------------------------------------------------
// wait
#include <alpaka/wait/Traits.hpp>
//-----------------------------------------------------------------------------
// workdiv
#include <alpaka/workdiv/Traits.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>
//-----------------------------------------------------------------------------
// vec
#include <alpaka/vec/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
