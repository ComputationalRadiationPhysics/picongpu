/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/event/EventCpu.hpp"
#include "alpaka/event/Traits.hpp"
#include "alpaka/kernel/TaskKernelCpuOmp2Blocks.hpp"
#include "alpaka/queue/QueueCpuBlocking.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/queue/cpu/ICpuQueue.hpp"
#include "alpaka/test/event/EventHostManualTrigger.hpp"
#include "alpaka/test/queue/Queue.hpp"
#include "alpaka/wait/Traits.hpp"

#include <atomic>
#include <mutex>

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

#    include <omp.h>

namespace alpaka
{
    namespace cpu::detail
    {
#    if BOOST_COMP_CLANG
// avoid diagnostic warning: "has no out-of-line virtual method definitions; its vtable will be emitted in every
// translation unit [-Werror,-Wweak-vtables]" https://stackoverflow.com/a/29288300
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wweak-vtables"
#    endif
        //! The CPU collective device queue implementation.
        struct QueueCpuOmp2CollectiveImpl final : cpu::ICpuQueue
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif
        {
            explicit QueueCpuOmp2CollectiveImpl(DevCpu const& dev) noexcept : m_dev(dev), blockingQueue(dev)
            {
            }

            void enqueue(EventCpu& ev) final
            {
                alpaka::enqueue(*this, ev);
            }

            void wait(EventCpu const& ev) final
            {
                alpaka::wait(*this, ev);
            }

            void busyWaitUntilBlockingQueueEmpty()
            {
                while(!empty(blockingQueue))
                    ;
            }

            DevCpu const m_dev; //!< The device this queue is bound to.
            std::mutex mutable m_mutex;
            QueueCpuBlocking blockingQueue;
            std::atomic<uint32_t> m_uCurrentlyExecutingTask = 0;
        };
    } // namespace cpu::detail

    //! The CPU collective device queue.
    //
    // @attention Queue can only be used together with the accelerator AccCpuOmp2Blocks.
    //
    // This queue is an example for a user provided queue and the behavior is strongly coupled
    // to the user workflows.
    //
    // Within an OpenMP parallel region kernel will be performed collectively.
    // All other operations will be performed from one thread (it is not defined which thread) and there will be no
    // implicit synchronization between other operations within the parallel OpenMP parallel region. Operations
    // executed within a OpenMP parallel region will be executed after already queued tasks before the parallel region
    // was created.
    //
    // Outside of an OpenMP parallel region the queue behaves like QueueCpuBlocking.
    struct QueueCpuOmp2Collective final : concepts::Implements<ConceptCurrentThreadWaitFor, QueueCpuOmp2Collective>
    {
        explicit QueueCpuOmp2Collective(DevCpu const& dev)
            : m_spQueueImpl(std::make_shared<cpu::detail::QueueCpuOmp2CollectiveImpl>(dev))
        {
            dev.registerQueue(m_spQueueImpl);
        }

        auto operator==(QueueCpuOmp2Collective const& rhs) const -> bool
        {
            return m_spQueueImpl == rhs.m_spQueueImpl;
        }

        auto operator!=(QueueCpuOmp2Collective const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

        std::shared_ptr<cpu::detail::QueueCpuOmp2CollectiveImpl> m_spQueueImpl;
    };

    namespace trait
    {
        //! The CPU blocking device queue device type trait specialization.
        template<>
        struct DevType<QueueCpuOmp2Collective>
        {
            using type = DevCpu;
        };

        //! The CPU blocking device queue device get trait specialization.
        template<>
        struct GetDev<QueueCpuOmp2Collective>
        {
            ALPAKA_FN_HOST static auto getDev(QueueCpuOmp2Collective const& queue) -> DevCpu
            {
                return queue.m_spQueueImpl->m_dev;
            }
        };

        //! The CPU blocking device queue event type trait specialization.
        template<>
        struct EventType<QueueCpuOmp2Collective>
        {
            using type = EventCpu;
        };

        //! The CPU blocking device queue enqueue trait specialization.
        //! This default implementation for all tasks directly invokes the function call operator of the task.
        template<typename TTask>
        struct Enqueue<QueueCpuOmp2Collective, TTask>
        {
            ALPAKA_FN_HOST static auto enqueue(QueueCpuOmp2Collective& queue, TTask const& task) -> void
            {
                if(::omp_in_parallel() != 0)
                {
                    // wait for all tasks enqueued before the parallel region
                    queue.m_spQueueImpl->busyWaitUntilBlockingQueueEmpty();
                    ++queue.m_spQueueImpl->m_uCurrentlyExecutingTask;
#    pragma omp single nowait
                    task();
                    --queue.m_spQueueImpl->m_uCurrentlyExecutingTask;
                }
                else
                {
                    std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                    alpaka::enqueue(queue.m_spQueueImpl->blockingQueue, task);
                }
            }
        };

        //! The CPU blocking device queue test trait specialization.
        template<>
        struct Empty<QueueCpuOmp2Collective>
        {
            ALPAKA_FN_HOST static auto empty(QueueCpuOmp2Collective const& queue) -> bool
            {
                return queue.m_spQueueImpl->m_uCurrentlyExecutingTask == 0u
                       && alpaka::empty(queue.m_spQueueImpl->blockingQueue);
            }
        };

        //! The CPU OpenMP2 collective device queue enqueue trait specialization.
        template<>
        struct Enqueue<cpu::detail::QueueCpuOmp2CollectiveImpl, EventCpu>
        {
            ALPAKA_FN_HOST static auto enqueue(cpu::detail::QueueCpuOmp2CollectiveImpl&, EventCpu&) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#    pragma omp barrier
            }
        };

        //! The CPU OpenMP2 collective device queue enqueue trait specialization.
        template<>
        struct Enqueue<QueueCpuOmp2Collective, EventCpu>
        {
            ALPAKA_FN_HOST static auto enqueue(QueueCpuOmp2Collective& queue, EventCpu& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(::omp_in_parallel() != 0)
                {
                    // wait for all tasks en-queued before the parallel region
                    queue.m_spQueueImpl->busyWaitUntilBlockingQueueEmpty();
#    pragma omp barrier
                }
                else
                {
                    alpaka::enqueue(queue.m_spQueueImpl->blockingQueue, event);
                }
            }
        };

        //! The CPU blocking device queue enqueue trait specialization.
        //! This default implementation for all tasks directly invokes the function call operator of the task.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct Enqueue<QueueCpuOmp2Collective, TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            ALPAKA_FN_HOST static auto enqueue(
                QueueCpuOmp2Collective& queue,
                TaskKernelCpuOmp2Blocks<TDim, TIdx, TKernelFnObj, TArgs...> const& task) -> void
            {
                if(::omp_in_parallel() != 0)
                {
                    queue.m_spQueueImpl->busyWaitUntilBlockingQueueEmpty();
                    ++queue.m_spQueueImpl->m_uCurrentlyExecutingTask;
                    // execute task within an OpenMP parallel region
                    task();
                    --queue.m_spQueueImpl->m_uCurrentlyExecutingTask;
                }
                else
                {
                    std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                    alpaka::enqueue(queue.m_spQueueImpl->blockingQueue, task);
                }
            }
        };

        template<>
        struct Enqueue<QueueCpuOmp2Collective, test::EventHostManualTriggerCpu<>>
        {
            ALPAKA_FN_HOST static auto enqueue(QueueCpuOmp2Collective&, test::EventHostManualTriggerCpu<>&) -> void
            {
                // EventHostManualTriggerCpu are not supported for together with the queue
                // QueueCpuOmp2Collective but a specialization is needed to path the EventTests
            }
        };

        //! The CPU blocking device queue thread wait trait specialization.
        //!
        //! Blocks execution of the calling thread until the queue has finished processing all previously requested
        //! tasks (kernels, data copies, ...)
        template<>
        struct CurrentThreadWaitFor<QueueCpuOmp2Collective>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(QueueCpuOmp2Collective const& queue) -> void
            {
                if(::omp_in_parallel() != 0)
                {
                    // wait for all tasks en-queued before the parallel region
                    queue.m_spQueueImpl->busyWaitUntilBlockingQueueEmpty();
#    pragma omp barrier
                }
                else
                {
                    std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                    wait(queue.m_spQueueImpl->blockingQueue);
                }
            }
        };

        //! The CPU OpenMP2 collective device queue event wait trait specialization.
        template<>
        struct WaiterWaitFor<cpu::detail::QueueCpuOmp2CollectiveImpl, EventCpu>
        {
            ALPAKA_FN_HOST static auto waiterWaitFor(cpu::detail::QueueCpuOmp2CollectiveImpl&, EventCpu const&) -> void
            {
#    pragma omp barrier
            }
        };

        //! The CPU OpenMP2 collective queue event wait trait specialization.
        template<>
        struct WaiterWaitFor<QueueCpuOmp2Collective, EventCpu>
        {
            ALPAKA_FN_HOST static auto waiterWaitFor(QueueCpuOmp2Collective& queue, EventCpu const& event) -> void
            {
                if(::omp_in_parallel() != 0)
                {
                    // wait for all tasks en-queued before the parallel region
                    queue.m_spQueueImpl->busyWaitUntilBlockingQueueEmpty();
                    wait(queue);
                }
                else
                    wait(queue.m_spQueueImpl->blockingQueue, event);
            }
        };
    } // namespace trait

    //! The blocking queue trait specialization for a OpenMP2 collective CPU queue.
    template<>
    struct test::trait::IsBlockingQueue<QueueCpuOmp2Collective> : std::true_type
    {
    };
} // namespace alpaka

#    include "alpaka/event/EventCpu.hpp"

#endif
