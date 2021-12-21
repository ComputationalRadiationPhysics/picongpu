/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#    if _OPENMP < 201307
#        error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#    endif

#    include <alpaka/core/Omp5.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/mem/buf/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/queue/Properties.hpp>
#    include <alpaka/queue/QueueGenericThreadsBlocking.hpp>
#    include <alpaka/queue/QueueGenericThreadsNonBlocking.hpp>
#    include <alpaka/queue/Traits.hpp>
#    include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>
#    include <alpaka/wait/Traits.hpp>

namespace alpaka
{
    class DevOmp5;
    namespace traits
    {
        template<typename TPltf, typename TSfinae>
        struct GetDevByIdx;
    }
    class PltfOmp5;

    namespace omp5
    {
        namespace detail
        {
            //#############################################################################
            //! The Omp5 device implementation.
            class DevOmp5Impl
            {
            public:
                //-----------------------------------------------------------------------------
                DevOmp5Impl(int iDevice) noexcept : m_iDevice(iDevice)
                {
                }
                //-----------------------------------------------------------------------------
                DevOmp5Impl(DevOmp5Impl const&) = delete;
                //-----------------------------------------------------------------------------
                DevOmp5Impl(DevOmp5Impl&&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(DevOmp5Impl const&) -> DevOmp5Impl& = delete;
                //-----------------------------------------------------------------------------
                auto operator=(DevOmp5Impl&&) -> DevOmp5Impl& = delete;
                //-----------------------------------------------------------------------------
                ~DevOmp5Impl() = default;

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto getAllExistingQueues() const
                    -> std::vector<std::shared_ptr<IGenericThreadsQueue<DevOmp5>>>
                {
                    std::vector<std::shared_ptr<IGenericThreadsQueue<DevOmp5>>> vspQueues;

                    std::lock_guard<std::mutex> lk(m_Mutex);
                    vspQueues.reserve(m_queues.size());

                    for(auto it = m_queues.begin(); it != m_queues.end();)
                    {
                        auto spQueue(it->lock());
                        if(spQueue)
                        {
                            vspQueues.emplace_back(std::move(spQueue));
                            ++it;
                        }
                        else
                        {
                            it = m_queues.erase(it);
                        }
                    }
                    return vspQueues;
                }

                //-----------------------------------------------------------------------------
                //! Registers the given queue on this device.
                //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
                ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<IGenericThreadsQueue<DevOmp5>> spQueue) -> void
                {
                    std::lock_guard<std::mutex> lk(m_Mutex);

                    // Register this queue on the device.
                    m_queues.push_back(spQueue);
                }

                int iDevice() const
                {
                    return m_iDevice;
                }

            private:
                std::mutex mutable m_Mutex;
                std::vector<std::weak_ptr<IGenericThreadsQueue<DevOmp5>>> mutable m_queues;
                const int m_iDevice;
            };
        } // namespace detail
    } // namespace omp5
    //#############################################################################
    //! The Omp5 device handle.
    class DevOmp5
        : public concepts::Implements<ConceptCurrentThreadWaitFor, DevOmp5>
        , public concepts::Implements<ConceptDev, DevOmp5>
    {
        friend struct traits::GetDevByIdx<PltfOmp5>;

        //-----------------------------------------------------------------------------
        DevOmp5(int iDevice) : m_spDevOmp5Impl(std::make_shared<omp5::detail::DevOmp5Impl>(iDevice))
        {
        }

    public:
        //-----------------------------------------------------------------------------
        DevOmp5(DevOmp5 const&) = default;
        //-----------------------------------------------------------------------------
        DevOmp5(DevOmp5&&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(DevOmp5 const&) -> DevOmp5& = default;
        //-----------------------------------------------------------------------------
        auto operator=(DevOmp5&&) -> DevOmp5& = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator==(DevOmp5 const& rhs) const -> bool
        {
            return m_spDevOmp5Impl->iDevice() == rhs.m_spDevOmp5Impl->iDevice();
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(DevOmp5 const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }
        //-----------------------------------------------------------------------------
        ~DevOmp5() = default;
        int iDevice() const
        {
            return m_spDevOmp5Impl->iDevice();
        }

        ALPAKA_FN_HOST auto getAllQueues() const -> std::vector<std::shared_ptr<IGenericThreadsQueue<DevOmp5>>>
        {
            return m_spDevOmp5Impl->getAllExistingQueues();
        }

        //-----------------------------------------------------------------------------
        //! Registers the given queue on this device.
        //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
        ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<IGenericThreadsQueue<DevOmp5>> spQueue) const -> void
        {
            m_spDevOmp5Impl->registerQueue(spQueue);
        }

    public:
        std::shared_ptr<omp5::detail::DevOmp5Impl> m_spDevOmp5Impl;
    };

    namespace traits
    {
        //#############################################################################
        //! The OpenMP 5.0 device name get trait specialization.
        template<>
        struct GetName<DevOmp5>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getName(DevOmp5 const&) -> std::string
            {
                return std::string("OMP5 target");
            }
        };

        //#############################################################################
        //! The OpenMP 5.0 device available memory get trait specialization.
        //!
        //! Returns 0, because querying target mem is not supported by OpenMP
        template<>
        struct GetMemBytes<DevOmp5>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getMemBytes(DevOmp5 const& dev) -> std::size_t
            {
                alpaka::ignore_unused(dev); //! \todo query device .. somehow

                return 0u;
            }
        };

        //#############################################################################
        //! The OpenMP 5.0 device free memory get trait specialization.
        //!
        //! Returns 0, because querying free target mem is not supported by OpenMP
        template<>
        struct GetFreeMemBytes<DevOmp5>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getFreeMemBytes(DevOmp5 const& dev) -> std::size_t
            {
                alpaka::ignore_unused(dev);

                return 0u;
            }
        };

        //#############################################################################
        //! The OpenMP 5.0 device warp size get trait specialization.
        template<>
        struct GetWarpSize<DevOmp5>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getWarpSize(DevOmp5 const& dev) -> std::size_t
            {
                alpaka::ignore_unused(dev);

                return 1u;
            }
        };

        //#############################################################################
        //! The OpenMP 5.0 device reset trait specialization.
        template<>
        struct Reset<DevOmp5>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto reset(DevOmp5 const& dev) -> void
            {
                alpaka::ignore_unused(dev); //! \TODO
            }
        };
    } // namespace traits

    template<typename TElem, typename TDim, typename TIdx>
    class BufOmp5;

    namespace traits
    {
        //#############################################################################
        //! The OpenMP 5.0 device memory buffer type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct BufType<DevOmp5, TElem, TDim, TIdx>
        {
            using type = BufOmp5<TElem, TDim, TIdx>;
        };

        //#############################################################################
        //! The OpenMP 5.0 device platform type trait specialization.
        template<>
        struct PltfType<DevOmp5>
        {
            using type = PltfOmp5;
        };
    } // namespace traits
    using QueueOmp5NonBlocking = QueueGenericThreadsNonBlocking<DevOmp5>;
    using QueueOmp5Blocking = QueueGenericThreadsBlocking<DevOmp5>;

    namespace traits
    {
        template<>
        struct QueueType<DevOmp5, Blocking>
        {
            using type = QueueOmp5Blocking;
        };

        template<>
        struct QueueType<DevOmp5, NonBlocking>
        {
            using type = QueueOmp5NonBlocking;
        };

        //#############################################################################
        //! The thread Omp5 device wait specialization.
        //!
        //! Blocks until the device has completed all preceding requested tasks.
        //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
        template<>
        struct CurrentThreadWaitFor<DevOmp5>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto currentThreadWaitFor(DevOmp5 const& dev) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                generic::currentThreadWaitForDevice(dev);
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
