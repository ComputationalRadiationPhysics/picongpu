/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#pragma once 

#include "mallocMC_utils.hpp"
#include "mallocMC_constraints.hpp"
#include "mallocMC_prefixes.hpp"

#include <boost/cstdint.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/mpl/bool.hpp>
#include <sstream>
#include <cassert>

#include <boost/mpl/assert.hpp>

namespace mallocMC{

  /**
   * @brief Defines Traits for certain Allocators
   *
   * This trait class provides information about the capabilities of the
   * allocator.
   * Available traits:
   * bool providesAvailableSlots: declares if the allocator implements a useful
   * version of getAvailableSlots().
   */
  template <class T>
  class  Traits : public T{
    public:
    static const bool providesAvailableSlots = T::CreationPolicy::providesAvailableSlots::value;
  };


  /**
   * @brief "HostClass" that combines all policies to a useful allocator
   *
   * This class implements the necessary glue-logic to form an actual allocator
   * from the provided policies. It implements the public interface and
   * executes some constraint checking based on an instance of the class
   * PolicyConstraints.
   *
   * @tparam T_CreationPolicy The desired type of a CreationPolicy
   * @tparam T_DistributionPolicy The desired type of a DistributionPolicy
   * @tparam T_OOMPolicy The desired type of a OOMPolicy
   * @tparam T_ReservePoolPolicy The desired type of a ReservePoolPolicy
   * @tparam T_AlignmentPolicy The desired type of a AlignmentPolicy
   */
  template < 
     typename T_CreationPolicy, 
     typename T_DistributionPolicy, 
     typename T_OOMPolicy, 
     typename T_ReservePoolPolicy,
     typename T_AlignmentPolicy
       >
  struct Allocator : 
    public T_CreationPolicy, 
    public T_OOMPolicy, 
    public T_ReservePoolPolicy,
    public T_AlignmentPolicy,
    public PolicyConstraints<T_CreationPolicy,T_DistributionPolicy,T_OOMPolicy,T_ReservePoolPolicy,T_AlignmentPolicy>
  {
    public:
      typedef T_CreationPolicy CreationPolicy;
      typedef T_DistributionPolicy DistributionPolicy;
      typedef T_OOMPolicy OOMPolicy;
      typedef T_ReservePoolPolicy ReservePoolPolicy;
      typedef T_AlignmentPolicy AlignmentPolicy;

    private:
      typedef boost::uint32_t uint32;
      void* pool;

    public:

      typedef Allocator<CreationPolicy,DistributionPolicy,
              OOMPolicy,ReservePoolPolicy,AlignmentPolicy> MyType;

      MAMC_ACCELERATOR
      void* alloc(size_t bytes){
        DistributionPolicy distributionPolicy;

        bytes            = AlignmentPolicy::applyPadding(bytes);
        uint32 req_size  = distributionPolicy.collect(bytes);
        void* memBlock   = CreationPolicy::create(req_size);
        const bool oom   = CreationPolicy::isOOM(memBlock, req_size);
        if(oom) memBlock = OOMPolicy::handleOOM(memBlock);
        void* myPart     = distributionPolicy.distribute(memBlock);

        return myPart;
        // if(blockIdx.x==0 && threadIdx.x==0){
        //     printf("warp %d trying to allocate %d bytes. myalloc: %p (oom %d)\n",GPUTools::warpid(),req_size,myalloc,oom);
        // }
      }

      MAMC_ACCELERATOR
      void dealloc(void* p){
        CreationPolicy::destroy(p);
      }

      MAMC_HOST
      void* initHeap(size_t size){
        pool = ReservePoolPolicy::setMemPool(size);
        boost::tie(pool,size) = AlignmentPolicy::alignPool(pool,size);
        void* h = CreationPolicy::initHeap(*this,pool,size);

        /*
        * This is a workaround for a bug with getAvailSlotsPoly:
        * Due to some problems with conditional compilation (possibly a CUDA bug),
        * this host function must explicitly be used from inside a host
        * function at least once. Doing it here guarantees that it is executed
        * and that this execution happens on the host. Usually, simply defining
        * this inside a host function (without actually executing it) would be
        * sufficient. However, due to the template nature of policy based
        * design, functions are only compiled if they are actually used.
        */
        if(CreationPolicy::providesAvailableSlots::value)
          CreationPolicy::getAvailableSlotsHost(1024,*this); //actual slot size does not matter

        return h;
      }

      MAMC_HOST
      void finalizeHeap(){
        CreationPolicy::finalizeHeap(*this,pool);
        ReservePoolPolicy::resetMemPool(pool);
      }

      MAMC_HOST
      static std::string info(std::string linebreak = " "){
        std::stringstream ss;
        ss << "CreationPolicy:      " << CreationPolicy::classname()     << linebreak;
        ss << "DistributionPolicy:  " << DistributionPolicy::classname() << linebreak;
        ss << "OOMPolicy:           " << OOMPolicy::classname()          << linebreak;
        ss << "ReservePoolPolicy:   " << ReservePoolPolicy::classname()  << linebreak;
        ss << "AlignmentPolicy:     " << AlignmentPolicy::classname()    << linebreak;
        return ss.str();
      }


      // polymorphism over the availability of getAvailableSlots
      MAMC_HOST MAMC_ACCELERATOR
      unsigned getAvailableSlots(size_t slotSize){
        return getAvailSlotsPoly(slotSize, boost::mpl::bool_<CreationPolicy::providesAvailableSlots::value>());
      }

    private:
      MAMC_HOST MAMC_ACCELERATOR
      unsigned getAvailSlotsPoly(size_t slotSize, boost::mpl::bool_<false>){
        return 0;
      }

      MAMC_HOST MAMC_ACCELERATOR
      unsigned getAvailSlotsPoly(size_t slotSize, boost::mpl::bool_<true>){
        slotSize = AlignmentPolicy::applyPadding(slotSize);
#ifdef __CUDA_ARCH__
        return CreationPolicy::getAvailableSlotsAccelerator(slotSize);
#else
        return CreationPolicy::getAvailableSlotsHost(slotSize,*this);
#endif
      }

  };

} //namespace mallocMC
