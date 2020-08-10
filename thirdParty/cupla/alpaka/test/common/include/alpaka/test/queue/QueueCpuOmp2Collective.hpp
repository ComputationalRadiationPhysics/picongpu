/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#if _OPENMP < 200203
    #error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#endif

#include <alpaka/test/queue/Queue.hpp>

#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/event/EventCpu.hpp>
#include <alpaka/queue/cpu/ICpuQueue.hpp>
#include <alpaka/queue/QueueCpuBlocking.hpp>
#include <alpaka/kernel/TaskKernelCpuOmp2Blocks.hpp>
#include <alpaka/test/event/EventHostManualTrigger.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <atomic>
#include <mutex>
#include <omp.h>

namespace alpaka
{
    namespace queue
    {
        namespace cpu
        {
            namespace detail
            {
#if BOOST_COMP_CLANG
    // avoid diagnostic warning: "has no out-of-line virtual method definitions; its vtable will be emitted in every translation unit [-Werror,-Wweak-vtables]"
    // https://stackoverflow.com/a/29288300
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wweak-vtables"
#endif
                //#############################################################################
                //! The CPU collective device queue implementation.
                class QueueCpuOmp2CollectiveImpl final : public cpu::ICpuQueue
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
                {
                public:
                    //-----------------------------------------------------------------------------
                    QueueCpuOmp2CollectiveImpl(
                        dev::DevCpu const & dev) noexcept :
                            m_dev(dev),
                            m_uCurrentlyExecutingTask(0u)
                    {}
                    //-----------------------------------------------------------------------------
                    QueueCpuOmp2CollectiveImpl(QueueCpuOmp2CollectiveImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueCpuOmp2CollectiveImpl(QueueCpuOmp2CollectiveImpl &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCpuOmp2CollectiveImpl const &) -> QueueCpuOmp2CollectiveImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueCpuOmp2CollectiveImpl &&) -> QueueCpuOmp2CollectiveImpl & = delete;
                    //-----------------------------------------------------------------------------
                    void enqueue(event::EventCpu & ev) final
                    {
                        queue::enqueue(*this, ev);
                    }
                    //-----------------------------------------------------------------------------
                    void wait(event::EventCpu const & ev) final
                    {
                        wait::wait(*this, ev);
                    }

                public:
                    dev::DevCpu const m_dev;            //!< The device this queue is bound to.
                    std::mutex mutable m_mutex;
                    std::atomic<uint32_t> m_uCurrentlyExecutingTask;
                };
            }
        }

        //#############################################################################
        //! The CPU collective device queue.
        //
        // @attention Queue can only be used together with the accelerator AccCpuOmp2Blocks.
        //
        // This queue is an example for a user provided queue and the behavior is strongly coupled
        // to the user workflows.
        //
        // Within a OpenMP parallel region kernel will be performed collectively.
        // All other operations will be performed from one thread (it is not defined which thread).
        //
        // Outside of a OpenMP parallel region the queue behaves like QueueCpuBlocking.
        class QueueCpuOmp2Collective final : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, QueueCpuOmp2Collective>
        {
        public:
            //-----------------------------------------------------------------------------
            QueueCpuOmp2Collective(
                dev::DevCpu const & dev) :
                    m_spQueueImpl(std::make_shared<cpu::detail::QueueCpuOmp2CollectiveImpl>(dev)),
                    m_spBlockingQueue(std::make_shared<QueueCpuBlocking>(dev))
            {
                dev.registerQueue(m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            QueueCpuOmp2Collective(QueueCpuOmp2Collective const &) = default;
            //-----------------------------------------------------------------------------
            QueueCpuOmp2Collective(QueueCpuOmp2Collective &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCpuOmp2Collective const &) -> QueueCpuOmp2Collective & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueCpuOmp2Collective &&) -> QueueCpuOmp2Collective & = default;
            //-----------------------------------------------------------------------------
            auto operator==(QueueCpuOmp2Collective const & rhs) const
            -> bool
            {
                return m_spQueueImpl == rhs.m_spQueueImpl && m_spBlockingQueue == rhs.m_spBlockingQueue;
            }
            //-----------------------------------------------------------------------------
            auto operator!=(QueueCpuOmp2Collective const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueCpuOmp2Collective() = default;

        public:
            std::shared_ptr<cpu::detail::QueueCpuOmp2CollectiveImpl> m_spQueueImpl;
            std::shared_ptr<QueueCpuBlocking> m_spBlockingQueue;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueCpuOmp2Collective>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU blocking device queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueCpuOmp2Collective>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueCpuOmp2Collective const & queue)
                -> dev::DevCpu
                {
                    return queue.m_spQueueImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue event type trait specialization.
            template<>
            struct EventType<
                queue::QueueCpuOmp2Collective>
            {
                using type = event::EventCpu;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {

            //#############################################################################
            //! The CPU blocking device queue enqueue trait specialization.
            //! This default implementation for all tasks directly invokes the function call operator of the task.
            template<
                typename TTask>
            struct Enqueue<
                queue::QueueCpuOmp2Collective,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCpuOmp2Collective & queue,
                    TTask const & task)
                -> void
                {
                    if(::omp_in_parallel() != 0)
                    {
                        // wait for all tasks en-queued before the parallel region
                        while(!queue::empty(*queue.m_spBlockingQueue)){}
                        queue.m_spQueueImpl->m_uCurrentlyExecutingTask += 1u;

                        #pragma omp single nowait
                        task();

                        queue.m_spQueueImpl->m_uCurrentlyExecutingTask -= 1u;
                    }
                    else
                    {
                        std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                        queue::enqueue(*queue.m_spBlockingQueue, task);
                    }
                }
            };

            //#############################################################################
            //! The CPU blocking device queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueCpuOmp2Collective>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueCpuOmp2Collective const & queue)
                -> bool
                {
                    return queue.m_spQueueImpl->m_uCurrentlyExecutingTask == 0u &&
                        queue::empty(*queue.m_spBlockingQueue);
                }
            };

            //#############################################################################
            //! The CPU OpenMP2 collective device queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::cpu::detail::QueueCpuOmp2CollectiveImpl,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::cpu::detail::QueueCpuOmp2CollectiveImpl &,
                    event::EventCpu &)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    #pragma omp barrier
                }
            };
            //#############################################################################
            //! The CPU OpenMP2 collective device queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueCpuOmp2Collective,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCpuOmp2Collective & queue,
                    event::EventCpu & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    if(::omp_in_parallel() != 0)
                    {
                        // wait for all tasks en-queued before the parallel region
                        while(!queue::empty(*queue.m_spBlockingQueue)){}
                        #pragma omp barrier
                    }
                    else
                    {
                        queue::enqueue(*queue.m_spBlockingQueue, event);
                    }

                }
            };

            //#############################################################################
            //! The CPU blocking device queue enqueue trait specialization.
            //! This default implementation for all tasks directly invokes the function call operator of the task.
            template<
                typename TDim,
                typename TIdx,
                typename TKernelFnObj,
                typename... TArgs>
            struct Enqueue<
                queue::QueueCpuOmp2Collective,
                kernel::TaskKernelCpuOmp2Blocks<
                    TDim,
                    TIdx,
                    TKernelFnObj,
                    TArgs...>>
            {
            private:
                using Task = kernel::TaskKernelCpuOmp2Blocks<
                    TDim,
                    TIdx,
                    TKernelFnObj,
                    TArgs ...>;
            public:
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCpuOmp2Collective & queue,
                    Task const & task)
                -> void
                {
                    if(::omp_in_parallel() != 0)
                    {
                        while(!queue::empty(*queue.m_spBlockingQueue)){}
                        // execute within an OpenMP parallel region
                        queue.m_spQueueImpl->m_uCurrentlyExecutingTask += 1u;
                        // execute task within an OpenMP parallel region
                        task();
                        queue.m_spQueueImpl->m_uCurrentlyExecutingTask -= 1u;
                    }
                    else
                    {
                        std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                        queue::enqueue(*queue.m_spBlockingQueue, task);
                    }
                }
            };

            //#############################################################################
            //!
            //#############################################################################
            template<>
            struct Enqueue<
                queue::QueueCpuOmp2Collective,
                test::event::EventHostManualTriggerCpu>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCpuOmp2Collective & ,
                    test::event::EventHostManualTriggerCpu & )
                -> void
                {
                    // EventHostManualTriggerCpu are not supported for together with the queue QueueCpuOmp2Collective
                    // but a specialization is needed to path the EventTests
                }
            };
        }
    }

    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU blocking device queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueCpuOmp2Collective>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueCpuOmp2Collective const & queue)
                -> void
                {
                    if(::omp_in_parallel() != 0)
                    {
                        // wait for all tasks en-queued before the parallel region
                        while(!queue::empty(*queue.m_spBlockingQueue)){}
                        #pragma omp barrier
                    }
                    else
                    {
                        std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
                        wait::wait(*queue.m_spBlockingQueue);
                    }
                }
            };


            //#############################################################################
            //! The CPU OpenMP2 collective device queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::cpu::detail::QueueCpuOmp2CollectiveImpl,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::cpu::detail::QueueCpuOmp2CollectiveImpl &,
                    event::EventCpu const &)
                -> void
                {
                    #pragma omp barrier
                }
            };
            //#############################################################################
            //! The CPU OpenMP2 collective queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueCpuOmp2Collective,
                event::EventCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueCpuOmp2Collective & queue,
                    event::EventCpu const & event)
                -> void
                {
                    if(::omp_in_parallel() != 0)
                    {
                        // wait for all tasks en-queued before the parallel region
                        while(!queue::empty(*queue.m_spBlockingQueue)){}
                        wait::wait(queue);
                    }
                    else
                        wait::wait(*queue.m_spBlockingQueue, event);
                }
            };
        }
    }
    //-----------------------------------------------------------------------------
    //! The test specifics.
    namespace test
    {
        //-----------------------------------------------------------------------------
        //! The test queue specifics.
        namespace queue
        {
            namespace traits
            {
                //#############################################################################
                //! The blocking queue trait specialization for a OpenMP2 collective CPU queue.
                template<>
                struct IsBlockingQueue<
                    alpaka::queue::QueueCpuOmp2Collective>
                {
                    static constexpr bool value = true;
                };
            }
        }
    }
}

#include <alpaka/event/EventCpu.hpp>

#endif
