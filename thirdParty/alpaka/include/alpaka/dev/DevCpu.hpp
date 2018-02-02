/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>

#include <alpaka/dev/cpu/SysInfo.hpp>

#include <boost/core/ignore_unused.hpp>

#include <map>
#include <mutex>
#include <memory>
#include <vector>
#include <algorithm>

namespace alpaka
{
    namespace stream
    {
        class StreamCpuAsync;

        namespace cpu
        {
            namespace detail
            {
                class StreamCpuAsyncImpl;
            }
        }
    }
    namespace pltf
    {
        namespace traits
        {
            template<
                typename TPltf,
                typename TSfinae>
            struct GetDevByIdx;
        }
        class PltfCpu;
    }
    namespace dev
    {
        //-----------------------------------------------------------------------------
        //! The CPU device.
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU device implementation.
                class DevCpuImpl
                {
                    friend stream::StreamCpuAsync;                   // stream::StreamCpuAsync::StreamCpuAsync calls RegisterAsyncStream.
                    friend stream::cpu::detail::StreamCpuAsyncImpl;  // StreamCpuAsyncImpl::~StreamCpuAsyncImpl calls UnregisterAsyncStream.
                public:
                    //-----------------------------------------------------------------------------
                    DevCpuImpl() = default;
                    //-----------------------------------------------------------------------------
                    DevCpuImpl(DevCpuImpl const &) = default;
                    //-----------------------------------------------------------------------------
                    DevCpuImpl(DevCpuImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(DevCpuImpl const &) -> DevCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    auto operator=(DevCpuImpl &&) -> DevCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    ~DevCpuImpl() = default;

                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto GetAllAsyncStreamImpls() const noexcept(false)
                    -> std::vector<std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl>>
                    {
                        std::vector<std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl>> vspStreams;

                        std::lock_guard<std::mutex> lk(m_Mutex);

                        for(auto const & pairStream : m_mapStreams)
                        {
                            auto spStream(pairStream.second.lock());
                            if(spStream)
                            {
                                vspStreams.emplace_back(std::move(spStream));
                            }
                            else
                            {
                                throw std::logic_error("One of the streams registered on the device is invalid!");
                            }
                        }
                        return vspStreams;
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //! Registers the given stream on this device.
                    //! NOTE: Every stream has to be registered for correct functionality of device wait operations!
                    ALPAKA_FN_HOST auto RegisterAsyncStream(std::shared_ptr<stream::cpu::detail::StreamCpuAsyncImpl> spStreamImpl)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Register this stream on the device.
                        // NOTE: We have to store the plain pointer next to the weak pointer.
                        // This is necessary to find the entry on unregistering because the weak pointer will already be invalid at that point.
                        m_mapStreams.emplace(spStreamImpl.get(), spStreamImpl);
                    }
                    //-----------------------------------------------------------------------------
                    //! Unregisters the given stream from this device.
                    ALPAKA_FN_HOST auto UnregisterAsyncStream(stream::cpu::detail::StreamCpuAsyncImpl const * const pStream) noexcept(false)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Unregister this stream from the device.
                        auto const itFind(std::find_if(
                            m_mapStreams.begin(),
                            m_mapStreams.end(),
                            [pStream](std::pair<stream::cpu::detail::StreamCpuAsyncImpl *, std::weak_ptr<stream::cpu::detail::StreamCpuAsyncImpl>> const & pair)
                            {
                                return (pStream == pair.first);
                            }));
                        if(itFind != m_mapStreams.end())
                        {
                            m_mapStreams.erase(itFind);
                        }
                        else
                        {
                            throw std::logic_error("The stream to unregister from the device could not be found in the list of registered streams!");
                        }
                    }

                private:
                    std::mutex mutable m_Mutex;
                    std::map<stream::cpu::detail::StreamCpuAsyncImpl *, std::weak_ptr<stream::cpu::detail::StreamCpuAsyncImpl>> m_mapStreams;
                };
            }
        }

        //#############################################################################
        //! The CPU device handle.
        class DevCpu
        {
            friend struct pltf::traits::GetDevByIdx<pltf::PltfCpu>;
        protected:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST DevCpu() :
                m_spDevCpuImpl(std::make_shared<cpu::detail::DevCpuImpl>())
            {}
        public:
            //-----------------------------------------------------------------------------
            DevCpu(DevCpu const &) = default;
            //-----------------------------------------------------------------------------
            DevCpu(DevCpu &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevCpu const &) -> DevCpu & = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevCpu &&) -> DevCpu & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(DevCpu const &) const
            -> bool
            {
                return true;
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(DevCpu const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~DevCpu() = default;

        public:
            std::shared_ptr<cpu::detail::DevCpuImpl> m_spDevCpuImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device name get trait specialization.
            template<>
            struct GetName<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevCpu const & dev)
                -> std::string
                {
                    boost::ignore_unused(dev);

                    return dev::cpu::detail::getCpuName();
                }
            };

            //#############################################################################
            //! The CPU device available memory get trait specialization.
            template<>
            struct GetMemBytes<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevCpu const & dev)
                -> std::size_t
                {
                    boost::ignore_unused(dev);

                    return dev::cpu::detail::getTotalGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The CPU device free memory get trait specialization.
            template<>
            struct GetFreeMemBytes<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevCpu const & dev)
                -> std::size_t
                {
                    boost::ignore_unused(dev);

                    return dev::cpu::detail::getFreeGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The CPU device reset trait specialization.
            template<>
            struct Reset<
                dev::DevCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevCpu const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    boost::ignore_unused(dev);

                    // The CPU does nothing on reset.
                }
            };
        }
    }
    namespace mem
    {
        namespace buf
        {
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            class BufCpu;

            namespace traits
            {
                //#############################################################################
                //! The CPU device memory buffer type trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct BufType<
                    dev::DevCpu,
                    TElem,
                    TDim,
                    TSize>
                {
                    using type = mem::buf::BufCpu<TElem, TDim, TSize>;
                };
            }
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device platform type trait specialization.
            template<>
            struct PltfType<
                dev::DevCpu>
            {
                using type = pltf::PltfCpu;
            };
        }
    }
}
