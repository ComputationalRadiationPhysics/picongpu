/* Copyright 2022 Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber, Antonio Di Pilato, Jeffrey Kelling
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
#    include <alpaka/dev/common/QueueRegistry.hpp>
#    include <alpaka/mem/buf/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/queue/Properties.hpp>
#    include <alpaka/queue/QueueGenericThreadsBlocking.hpp>
#    include <alpaka/queue/QueueGenericThreadsNonBlocking.hpp>
#    include <alpaka/queue/Traits.hpp>
#    include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>
#    include <alpaka/traits/Traits.hpp>
#    include <alpaka/wait/Traits.hpp>

#    include <cstddef>
#    include <map>
#    include <sstream>
#    include <stdexcept>
#    include <string>
#    include <vector>

namespace alpaka
{
    class DevOmp5;
    namespace trait
    {
        template<typename TPltf, typename TSfinae>
        struct GetDevByIdx;
    }
    class PltfOmp5;

    namespace omp5::detail
    {
        //! The Omp5 device implementation.
        class DevOmp5Impl : public alpaka::detail::QueueRegistry<IGenericThreadsQueue<DevOmp5>>
        {
        public:
            DevOmp5Impl(int iDevice) noexcept : m_iDevice(iDevice)
            {
            }
            ~DevOmp5Impl()
            {
                for(auto& a : m_staticMemMap)
                    omp_target_free(a.second.first, getNativeHandle());
            }

            [[nodiscard]] auto getNativeHandle() const noexcept -> int
            {
                return m_iDevice;
            }

            //! Create and/or return staticlly mapped device pointer of host address.
            template<typename TElem, typename TExtent>
            ALPAKA_FN_HOST auto mapStatic(TElem* pHost, TExtent const& extent) -> TElem*
            {
                const std::size_t sizeB = extent.prod() * sizeof(TElem);
                auto m = m_staticMemMap.find(pHost);
                if(m != m_staticMemMap.end())
                {
                    if(sizeB != m->second.second)
                    {
                        std::ostringstream os;
                        os << "Statically mapped size cannot change: Static size is " << m->second.second
                           << " requested size is " << sizeB << '.';
                        throw std::runtime_error(os.str());
                    }
                    return reinterpret_cast<TElem*>(m->second.first);
                }

                void* pDev = omp_target_alloc(sizeB, getNativeHandle());
                if(!pDev)
                    return nullptr;

                /*! Associating pointers for good measure. Not actually
                 * required as long a not `target enter data` is done */
                omp_target_associate_ptr(pHost, pDev, sizeB, 0u, getNativeHandle());

                m_staticMemMap[pHost] = std::make_pair(pDev, sizeB);
                return reinterpret_cast<TElem*>(pDev);
            }

        private:
            int const m_iDevice;

            std::map<void*, std::pair<void*, std::size_t>> m_staticMemMap;
        };
    } // namespace omp5::detail
    //! The Omp5 device handle.
    class DevOmp5
        : public concepts::Implements<ConceptCurrentThreadWaitFor, DevOmp5>
        , public concepts::Implements<ConceptDev, DevOmp5>
    {
        friend struct trait::GetDevByIdx<PltfOmp5>;

        DevOmp5(int iDevice) : m_spDevOmp5Impl(std::make_shared<omp5::detail::DevOmp5Impl>(iDevice))
        {
        }

    public:
        ALPAKA_FN_HOST auto operator==(DevOmp5 const& rhs) const -> bool
        {
            return getNativeHandle() == rhs.getNativeHandle();
        }
        ALPAKA_FN_HOST auto operator!=(DevOmp5 const& rhs) const -> bool
        {
            return !((*this) == rhs);
        }

        [[nodiscard]] auto getNativeHandle() const noexcept -> int
        {
            return m_spDevOmp5Impl->getNativeHandle();
        }

        //! Create and/or return staticlly mapped device pointer of host address.
        template<typename TElem, typename TExtent>
        ALPAKA_FN_HOST auto mapStatic(TElem* pHost, TExtent const& extent) const -> TElem*
        {
            return m_spDevOmp5Impl->mapStatic(pHost, extent);
        }

        [[nodiscard]] ALPAKA_FN_HOST auto getAllQueues() const
            -> std::vector<std::shared_ptr<IGenericThreadsQueue<DevOmp5>>>
        {
            return m_spDevOmp5Impl->getAllExistingQueues();
        }

        //! Registers the given queue on this device.
        //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
        ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<IGenericThreadsQueue<DevOmp5>> spQueue) const -> void
        {
            m_spDevOmp5Impl->registerQueue(spQueue);
        }

        auto registerCleanup(omp5::detail::DevOmp5Impl::CleanerFunctor c) const -> void
        {
            m_spDevOmp5Impl->registerCleanup(c);
        }

    public:
        std::shared_ptr<omp5::detail::DevOmp5Impl> m_spDevOmp5Impl;
    };

    namespace trait
    {
        //! The OpenMP 5.0 device name get trait specialization.
        template<>
        struct GetName<DevOmp5>
        {
            ALPAKA_FN_HOST static auto getName(DevOmp5 const&) -> std::string
            {
                return std::string("OMP5 target");
            }
        };

        //! The OpenMP 5.0 device available memory get trait specialization.
        //!
        //! Returns 0, because querying target mem is not supported by OpenMP
        template<>
        struct GetMemBytes<DevOmp5>
        {
            ALPAKA_FN_HOST static auto getMemBytes(DevOmp5 const& /* dev */) -> std::size_t
            {
                //! \todo query device .. somehow
                return 0u;
            }
        };

        //! The OpenMP 5.0 device free memory get trait specialization.
        //!
        //! Returns 0, because querying free target mem is not supported by OpenMP
        template<>
        struct GetFreeMemBytes<DevOmp5>
        {
            ALPAKA_FN_HOST static auto getFreeMemBytes(DevOmp5 const& /* dev */) -> std::size_t
            {
                return 0u;
            }
        };

        //! The OpenMP 5.0 device warp size get trait specialization.
        template<>
        struct GetWarpSizes<DevOmp5>
        {
            ALPAKA_FN_HOST static auto getWarpSizes(DevOmp5 const& /* dev */) -> std::vector<std::size_t>
            {
                return {1u};
            }
        };

        //! The OpenMP 5.0 device reset trait specialization.
        template<>
        struct Reset<DevOmp5>
        {
            ALPAKA_FN_HOST static auto reset(DevOmp5 const& /* dev */) -> void
            {
                //! \TODO
            }
        };

        //! The OpenMP 5.0 device native handle trait specialization.
        template<>
        struct NativeHandle<DevOmp5>
        {
            [[nodiscard]] static auto getNativeHandle(DevOmp5 const& dev)
            {
                return dev.getNativeHandle();
            }
        };
    } // namespace trait

    template<typename TElem, typename TDim, typename TIdx>
    class BufOmp5;

    namespace trait
    {
        //! The OpenMP 5.0 device memory buffer type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct BufType<DevOmp5, TElem, TDim, TIdx>
        {
            using type = BufOmp5<TElem, TDim, TIdx>;
        };

        //! The OpenMP 5.0 device platform type trait specialization.
        template<>
        struct PltfType<DevOmp5>
        {
            using type = PltfOmp5;
        };
    } // namespace trait
    using QueueOmp5NonBlocking = QueueGenericThreadsNonBlocking<DevOmp5>;
    using QueueOmp5Blocking = QueueGenericThreadsBlocking<DevOmp5>;

    namespace trait
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

        //! The thread Omp5 device wait specialization.
        //!
        //! Blocks until the device has completed all preceding requested tasks.
        //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
        template<>
        struct CurrentThreadWaitFor<DevOmp5>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(DevOmp5 const& dev) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                generic::currentThreadWaitForDevice(dev);
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
