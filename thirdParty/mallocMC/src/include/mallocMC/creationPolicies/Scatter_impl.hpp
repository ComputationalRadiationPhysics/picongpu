/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014-2016 Institute of Radiation Physics,
                          Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Rene Widera - r.widera ( at ) hzdr.de
              Axel Huebl - a.huebl ( at ) hzdr.de
              Carlchristian Eckert - c.eckert ( at ) hzdr.de

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

#include <cstdio>
#include <boost/cstdint.hpp> /* uint32_t */
#include <iostream>
#include <string>
#include <cassert>
#include <stdexcept>
#include <boost/mpl/bool.hpp>

#include "../mallocMC_utils.hpp"
#include "Scatter.hpp"

namespace mallocMC{
namespace CreationPolicies{

namespace ScatterKernelDetail{
  template <typename T_Allocator>
  __global__ void initKernel(T_Allocator* heap, void* heapmem, size_t memsize){
    heap->pool = heapmem;
    heap->initDeviceFunction(heapmem, memsize);
  }


  template < typename T_Allocator >
  __global__ void getAvailableSlotsKernel(T_Allocator* heap, size_t slotSize, unsigned* slots){
    int gid       = threadIdx.x + blockIdx.x*blockDim.x;
    int nWorker   = gridDim.x * blockDim.x;
    unsigned temp = heap->getAvailaibleSlotsDeviceFunction(slotSize, gid, nWorker);
    if(temp) atomicAdd(slots, temp);
  }


  template <typename T_Allocator>
  __global__ void finalizeKernel(T_Allocator* heap){
    heap->finalizeDeviceFunction();
  }

} //namespace ScatterKernelDetail

  template<class T_Config, class T_Hashing>
  class Scatter
  {

    public:
      typedef T_Config  HeapProperties;
      typedef T_Hashing HashingProperties;
      struct  Properties : HeapProperties, HashingProperties{};
      typedef boost::mpl::bool_<true>  providesAvailableSlots;

    private:
      typedef boost::uint32_t uint32;


/** Allow for a hierarchical validation of parameters:
 *
 * shipped default-parameters (in the inherited struct) have lowest precedence.
 * They will be overridden by a given configuration struct. However, even the
 * given configuration struct can be overridden by compile-time command line
 * parameters (e.g. -D MALLOCMC_CP_SCATTER_PAGESIZE 1024)
 *
 * default-struct < template-struct < command-line parameter
 */
#ifndef MALLOCMC_CP_SCATTER_PAGESIZE
#define MALLOCMC_CP_SCATTER_PAGESIZE  static_cast<uint32>(HeapProperties::pagesize::value)
#endif
      BOOST_STATIC_CONSTEXPR uint32 pagesize      = MALLOCMC_CP_SCATTER_PAGESIZE;

#ifndef MALLOCMC_CP_SCATTER_ACCESSBLOCKS
#define MALLOCMC_CP_SCATTER_ACCESSBLOCKS static_cast<uint32>(HeapProperties::accessblocks::value)
#endif
      BOOST_STATIC_CONSTEXPR uint32 accessblocks  = MALLOCMC_CP_SCATTER_ACCESSBLOCKS;

#ifndef MALLOCMC_CP_SCATTER_REGIONSIZE
#define MALLOCMC_CP_SCATTER_REGIONSIZE static_cast<uint32>(HeapProperties::regionsize::value)
#endif
      BOOST_STATIC_CONSTEXPR uint32 regionsize    = MALLOCMC_CP_SCATTER_REGIONSIZE;

#ifndef MALLOCMC_CP_SCATTER_WASTEFACTOR
#define MALLOCMC_CP_SCATTER_WASTEFACTOR static_cast<uint32>(HeapProperties::wastefactor::value)
#endif
      BOOST_STATIC_CONSTEXPR uint32 wastefactor   = MALLOCMC_CP_SCATTER_WASTEFACTOR;

#ifndef MALLOCMC_CP_SCATTER_RESETFREEDPAGES
#define MALLOCMC_CP_SCATTER_RESETFREEDPAGES static_cast<bool>(HeapProperties::resetfreedpages::value)
#endif
      BOOST_STATIC_CONSTEXPR bool resetfreedpages = MALLOCMC_CP_SCATTER_RESETFREEDPAGES;


    public:
      BOOST_STATIC_CONSTEXPR uint32 _pagesize       = pagesize;
      BOOST_STATIC_CONSTEXPR uint32 _accessblocks   = accessblocks;
      BOOST_STATIC_CONSTEXPR uint32 _regionsize     = regionsize;
      BOOST_STATIC_CONSTEXPR uint32 _wastefactor    = wastefactor;
      BOOST_STATIC_CONSTEXPR bool _resetfreedpages  = resetfreedpages;

    private:
#if _DEBUG || ANALYSEHEAP
    public:
#endif
      //BOOST_STATIC_CONSTEXPR uint32 minChunkSize0 = pagesize/(32*32);
      BOOST_STATIC_CONSTEXPR uint32 minChunkSize1 = 0x10;
      BOOST_STATIC_CONSTEXPR uint32 HierarchyThreshold =  (pagesize - 2*sizeof(uint32))/33;
      BOOST_STATIC_CONSTEXPR uint32 minSegmentSize = 32*minChunkSize1 + sizeof(uint32);
      BOOST_STATIC_CONSTEXPR uint32 tmp_maxOPM = minChunkSize1 > HierarchyThreshold ? 0 : (pagesize + (minSegmentSize-1)) / minSegmentSize;
      BOOST_STATIC_CONSTEXPR uint32 maxOnPageMasks = 32 > tmp_maxOPM ? tmp_maxOPM : 32;

#ifndef MALLOCMC_CP_SCATTER_HASHINGK
#define MALLOCMC_CP_SCATTER_HASHINGK    static_cast<uint32>(HashingProperties::hashingK::value)
#endif
     BOOST_STATIC_CONSTEXPR uint32 hashingK       = MALLOCMC_CP_SCATTER_HASHINGK;

#ifndef MALLOCMC_CP_SCATTER_HASHINGDISTMP
#define MALLOCMC_CP_SCATTER_HASHINGDISTMP static_cast<uint32>(HashingProperties::hashingDistMP::value)
#endif
     BOOST_STATIC_CONSTEXPR uint32 hashingDistMP  = MALLOCMC_CP_SCATTER_HASHINGDISTMP;

#ifndef MALLOCMC_CP_SCATTER_HASHINGDISTWP
#define MALLOCMC_CP_SCATTER_HASHINGDISTWP static_cast<uint32>(HashingProperties::hashingDistWP::value)
#endif
     BOOST_STATIC_CONSTEXPR uint32 hashingDistWP  = MALLOCMC_CP_SCATTER_HASHINGDISTWP;

#ifndef MALLOCMC_CP_SCATTER_HASHINGDISTWPREL
#define MALLOCMC_CP_SCATTER_HASHINGDISTWPREL static_cast<uint32>(HashingProperties::hashingDistWPRel::value)
#endif
     BOOST_STATIC_CONSTEXPR uint32 hashingDistWPRel = MALLOCMC_CP_SCATTER_HASHINGDISTWPREL;


      /**
       * Page Table Entry struct
       * The PTE holds basic information about each page
       */
      struct PTE
      {
        uint32 chunksize;
        uint32 count;
        uint32 bitmask;

        __device__ void init()
        {
          chunksize = 0;
          count = 0;
          bitmask = 0;
        }
      };

      /**
       * Page struct
       * The page struct is used to access the data on the page more efficiently
       * and to clear the area on the page, which might hold bitsfields later one
       */
      struct PAGE
      {
        char data[pagesize];

        /**
         * The pages init method
         * This method initializes the region on the page which might hold
         * bit fields when the page is used for a small chunk size
         * @param previous_chunksize the chunksize which was uses for the page before
         */
        __device__ void init()
        {
          //clear the entire data which can hold bitfields
          uint32* write = (uint32*)(data + pagesize - (int)(sizeof(uint32)*maxOnPageMasks));
          while(write < (uint32*)(data + pagesize))
            *write++ = 0;
        }
      };

      // the data used by the allocator

      volatile PTE* _ptes;
      volatile uint32* _regions;
      PAGE* _page;
      uint32 _numpages;
      size_t _memsize;
      uint32 _pagebasedMutex;
      volatile uint32 _firstFreePageBased;
      volatile uint32 _firstfreeblock;


      /**
       * randInit should create an random offset which can be used
       * as the initial position in a bitfield
       */
      __device__ inline uint32 randInit()
      {
        //start with the laneid offset
        return laneid();
      }

      /**
       * randInextspot delivers the next free spot in a bitfield
       * it searches for the next unset bit to the left of spot and
       * returns its offset. if there are no unset bits to the left
       * then it wraps around
       * @param bitfield the bitfield to be searched for
       * @param spot the spot from which to search to the left
       * @param spots number of bits that can be used
       * @return next free spot in the bitfield
       */
      __device__ inline uint32 nextspot(uint32 bitfield, uint32 spot, uint32 spots)
      {
        //wrap around the bitfields from the current spot to the left
        bitfield = ((bitfield >> (spot + 1)) | (bitfield << (spots - (spot + 1))))&((1<<spots)-1);
        //compute the step from the current spot in the bitfield
        uint32 step = __ffs(~bitfield);
        //and return the new spot
        return (spot + step) % spots;
      }


      /**
       * onPageMasksPosition returns a pointer to the beginning of the onpagemasks inside a page.
       * @param page the page that holds the masks
       * @param the number of hierarchical page tables (bitfields) that are used inside this mask.
       * @return pointer to the first address inside the page that holds metadata bitfields.
       */
      __device__ inline uint32* onPageMasksPosition(uint32 page, uint32 nMasks){
        return (uint32*)(_page[page].data + pagesize - (int)sizeof(uint32)*nMasks);
      }

      /**
       * usespot marks finds one free spot in the bitfield, marks it and returns its offset
       * @param bitfield pointer to the bitfield to use
       * @param spots overall number of spots the bitfield is responsible for
       * @return if there is a free spot it returns the spot'S offset, otherwise -1
       */
      __device__ inline int usespot(uint32 *bitfield, uint32 spots)
      {
        //get first spot
        uint32 spot = randInit() % spots;
        for(;;)
        {
          uint32 mask = 1 << spot;
          uint32 old = atomicOr(bitfield, mask);
          if( (old & mask) == 0)
            return spot;
          // note: __popc(old) == spots should be sufficient,
          //but if someone corrupts the memory we end up in an endless loop in here...
          if(__popc(old) >= spots)
            return -1;
          spot = nextspot(old, spot, spots);
        }
      }


      /**
       * calcAdditionalChunks determines the number of chunks that are contained in the last segment of a hierarchical page
       *
       * The additional checks are necessary to ensure correct results for very large pages and small chunksizes
       *
       * @param fullsegments the number of segments that can be completely filled in a page. This may NEVER be bigger than 32!
       * @param segmentsize the number of bytes that are contained in a completely filled segment (32 chunks)
       * @param chunksize the chosen allocation size within the page
       * @return the number of additional chunks that will not fit in one of the fullsegments. For any correct input, this number is smaller than 32
       */
      __device__ inline uint32 calcAdditionalChunks(uint32 fullsegments, uint32 segmentsize, uint32 chunksize){
        if(fullsegments != 32){
          return max(0,(int)pagesize - (int)fullsegments*segmentsize - (int)sizeof(uint32))/chunksize;
        }else
          return 0;
      }


      /**
       * addChunkHierarchy finds a free chunk on a page which uses bit fields on the page
       * @param chunksize the chunksize of the page
       * @param fullsegments the number of full segments on the page (a 32 bits on the page)
       * @param additional_chunks the number of additional chunks in last segment (less than 32 bits on the page)
       * @param page the page to use
       * @return pointer to a free chunk on the page, 0 if we were unable to obtain a free chunk
       */
      __device__ inline void* addChunkHierarchy(uint32 chunksize, uint32 fullsegments, uint32 additional_chunks, uint32 page)
      {
        uint32 segments = fullsegments + (additional_chunks > 0 ? 1 : 0);
        uint32 spot = randInit() % segments;
        uint32 mask = _ptes[page].bitmask;
        if((mask & (1 << spot)) != 0)
          spot = nextspot(mask, spot, segments);
        uint32 tries = segments - __popc(mask);
        uint32* onpagemasks = onPageMasksPosition(page,segments);
        for(uint32 i = 0; i < tries; ++i)
        {
          int hspot = usespot(onpagemasks + spot, spot < fullsegments ? 32 : additional_chunks);
          if(hspot != -1)
            return _page[page].data + (32*spot + hspot)*chunksize;
          else
            atomicOr((uint32*)&_ptes[page].bitmask, 1 << spot);
          spot = nextspot(mask, spot, segments);
        }
        return 0;
      }

      /**
       * addChunkNoHierarchy finds a free chunk on a page which uses the bit fields of the pte only
       * @param chunksize the chunksize of the page
       * @param page the page to use
       * @param spots the number of chunks which fit on the page
       * @return pointer to a free chunk on the page, 0 if we were unable to obtain a free chunk
       */
      __device__ inline void* addChunkNoHierarchy(uint32 chunksize, uint32 page, uint32 spots)
      {
        int spot = usespot((uint32*)&_ptes[page].bitmask, spots);
        if(spot == -1)
          return 0; //that should be impossible :)
        return _page[page].data + spot*chunksize;
      }

      /**
       * tryUsePage tries to use the page for the allocation request
       * @param page the page to use
       * @param chunksize the chunksize of the page
       * @return pointer to a free chunk on the page, 0 if we were unable to obtain a free chunk
       */
      __device__ inline void* tryUsePage(uint32 page, uint32 chunksize)
      {

        void* chunk_ptr = NULL;

        //increse the fill level
        uint32 filllevel = atomicAdd((uint32*)&(_ptes[page].count), 1);
        //recheck chunck size (it could be that the page got freed in the meanwhile...)
        if(!resetfreedpages || _ptes[page].chunksize == chunksize)
        {
          if(chunksize <= HierarchyThreshold)
          {
            //more chunks than can be covered by the pte's single bitfield can be used
            uint32 segmentsize = chunksize*32 + sizeof(uint32);
            uint32 fullsegments = min(32,pagesize / segmentsize);
            uint32 additional_chunks = calcAdditionalChunks(fullsegments, segmentsize, chunksize);
            if(filllevel < fullsegments * 32 + additional_chunks)
              chunk_ptr = addChunkHierarchy(chunksize, fullsegments, additional_chunks, page);
          }
          else
          {
            uint32 chunksinpage = min(pagesize / chunksize, 32);
            if(filllevel < chunksinpage)
              chunk_ptr = addChunkNoHierarchy(chunksize, page, chunksinpage);
          }
        }

        //this one is full/not useable
        if(chunk_ptr == NULL)
          atomicSub((uint32*)&(_ptes[page].count), 1);

        return chunk_ptr;
      }


      /**
       * allocChunked tries to allocate the demanded number of bytes on one of the pages
       * @param bytes the number of bytes to allocate
       * @return pointer to a free chunk on a page, 0 if we were unable to obtain a free chunk
       */
      __device__ void* allocChunked(uint32 bytes)
      {
        uint32 pagesperblock = _numpages/accessblocks;
        uint32 reloff = warpSize*bytes / pagesize;
        uint32 startpage = (bytes*hashingK + hashingDistMP*smid() + (hashingDistWP+hashingDistWPRel*reloff)*warpid() ) % pagesperblock;
        uint32 maxchunksize = min(pagesize,wastefactor*bytes);
        uint32 startblock = _firstfreeblock;
        uint32 ptetry = startpage + startblock*pagesperblock;
        uint32 checklevel = regionsize*3/4;
        for(uint32 finder = 0; finder < 2; ++finder)
        {
          for(uint32 b = startblock; b < accessblocks; ++b)
          {
            while(ptetry < (b+1)*pagesperblock)
            {
              uint32 region = ptetry/regionsize;
              uint32 regionfilllevel = _regions[region];
              if(regionfilllevel < checklevel )
              {
                for( ; ptetry < (region+1)*regionsize; ++ptetry)
                {
                  uint32 chunksize = _ptes[ptetry].chunksize;
                  if(chunksize >= bytes && chunksize <= maxchunksize)
                  {
                    void * res = tryUsePage(ptetry, chunksize);
                    if(res != 0)  return res;
                  }
                  else if(chunksize == 0)
                  {
                    //lets open up a new page
                    //it is already padded
                    uint32 new_chunksize = max(bytes,minChunkSize1);
                    uint32 beforechunksize = atomicCAS((uint32*)&_ptes[ptetry].chunksize, 0, new_chunksize);
                    if(beforechunksize == 0)
                    {
                      void * res = tryUsePage(ptetry, new_chunksize);
                      if(res != 0)  return res;
                    }
                    else if(beforechunksize >= bytes &&  beforechunksize <= maxchunksize)
                    {
                      //someone else aquired the page, but we can also use it
                      void * res = tryUsePage(ptetry, beforechunksize);
                      if(res != 0)  return res;
                    }
                  }
                }
                //could not alloc in region, tell that
                if(regionfilllevel + 1 <= regionsize)
                  atomicMax((uint32*)(_regions + region), regionfilllevel+1);
              }
              else
                ptetry += regionsize;
              //ptetry = (region+1)*regionsize;
            }
            //randomize the thread writing the info
            //if(warpid() + laneid() == 0)
            if(b > startblock)
              _firstfreeblock = b;
          }

          //we are really full :/ so lets search every page for a spot!
          startblock = 0;
          checklevel = regionsize + 1;
          ptetry = 0;
        }
        return 0;
      }


      /**
       * deallocChunked frees the chunk on the page and updates all data accordingly
       * @param mem pointer to the chunk
       * @param page the page the chunk is on
       * @param chunksize the chunksize used for the page
       */
      __device__ void deallocChunked(void* mem, uint32 page, uint32 chunksize)
      {
        uint32 inpage_offset = ((char*)mem - _page[page].data);
        if(chunksize <= HierarchyThreshold)
        {
          //one more level in hierarchy
          uint32 segmentsize = chunksize*32 + sizeof(uint32);
          uint32 fullsegments = min(32,pagesize / segmentsize);
          uint32 additional_chunks = calcAdditionalChunks(fullsegments,segmentsize,chunksize);
          uint32 segment = inpage_offset / (chunksize*32);
          uint32 withinsegment = (inpage_offset - segment*(chunksize*32))/chunksize;
          //mark it as free
          uint32 nMasks = fullsegments + (additional_chunks > 0 ? 1 : 0);
          uint32* onpagemasks = onPageMasksPosition(page,nMasks);
          uint32 old = atomicAnd(onpagemasks + segment, ~(1 << withinsegment));

          // always do this, since it might fail due to a race-condition with addChunkHierarchy
          atomicAnd((uint32*)&_ptes[page].bitmask, ~(1 << segment));
        }
        else
        {
          uint32 segment = inpage_offset / chunksize;
          atomicAnd((uint32*)&_ptes[page].bitmask, ~(1 << segment));
        }
        //reduce filllevel as free
        uint32 oldfilllevel = atomicSub((uint32*)&_ptes[page].count, 1);


        if(resetfreedpages)
        {
          if(oldfilllevel == 1)
          {
            //this page now got free!
            // -> try lock it
            uint32 old = atomicCAS((uint32*)&_ptes[page].count, 0, pagesize);
            if(old == 0)
            {
              //clean the bits for the hierarchy
              _page[page].init();
              //remove chunk information
              _ptes[page].chunksize = 0;
              __threadfence();
              //unlock it
              atomicSub((uint32*)&_ptes[page].count, pagesize);
            }
          }
        }

        //meta information counters ... should not be changed by too many threads, so..
        if(oldfilllevel == pagesize / 2 / chunksize)
        {
          uint32 region = page / regionsize;
          _regions[region] = 0;
          uint32 block = region * regionsize * accessblocks / _numpages ;
          if(warpid() + laneid() == 0)
            atomicMin((uint32*)&_firstfreeblock, block);
        }
      }

      /**
       * markpages markes a fixed number of pages as used
       * @param startpage first page to mark
       * @param pages number of pages to mark
       * @param bytes number of overall bytes to mark pages for
       * @return true on success, false if one of the pages is not free
       */
      __device__ bool markpages(uint32 startpage, uint32 pages, uint32 bytes)
      {
        int abord = -1;
        for(uint32 trypage = startpage; trypage < startpage + pages; ++trypage)
        {
          uint32 old = atomicCAS((uint32*)&_ptes[trypage].chunksize, 0, bytes);
          if(old != 0)
          {
            abord = trypage;
            break;
          }
        }
        if(abord == -1)
          return true;
        for(uint32 trypage = startpage; trypage < abord; ++trypage)
          atomicCAS((uint32*)&_ptes[trypage].chunksize, bytes, 0);
        return false;
      }

      /**
       * allocPageBasedSingleRegion tries to allocate the demanded number of bytes on a continues sequence of pages
       * @param startpage first page to be used
       * @param endpage last page to be used
       * @param bytes number of overall bytes to mark pages for
       * @return pointer to the first page to use, 0 if we were unable to use all the requested pages
       */
      __device__ void* allocPageBasedSingleRegion(uint32 startpage, uint32 endpage, uint32 bytes)
      {
        uint32 pagestoalloc = divup(bytes, pagesize);
        uint32 freecount = 0;
        bool left_free = false;
        for(uint32 search_page = startpage+1; search_page > endpage; )
        {
          --search_page;
          if(_ptes[search_page].chunksize == 0)
          {
            if(++freecount == pagestoalloc)
            {
              //try filling it up
              if(markpages(search_page, pagestoalloc, bytes))
              {
                //mark that we filled up everything up to here
                if(!left_free)
                  atomicCAS((uint32*)&_firstFreePageBased, startpage, search_page - 1);
                return _page[search_page].data;
              }
            }
          }
          else
          {
            left_free = true;
            freecount = 0;
          }
        }
        return 0;
      }

      /**
       * allocPageBasedSingle tries to allocate the demanded number of bytes on a continues sequence of pages
       * @param bytes number of overall bytes to mark pages for
       * @return pointer to the first page to use, 0 if we were unable to use all the requested pages
       * @pre only a single thread of a warp is allowed to call the function concurrently
       */
      __device__ void* allocPageBasedSingle(uint32 bytes)
      {
        //acquire mutex
        while(atomicExch(&_pagebasedMutex,1) != 0);
        //search for free spot from the back
        uint32 spage = _firstFreePageBased;
        void* res = allocPageBasedSingleRegion(spage, 0, bytes);
        if(res == 0)
          //also check the rest of the pages
          res = allocPageBasedSingleRegion(_numpages, spage, bytes);

        //free mutex
        atomicExch(&_pagebasedMutex,0);
        return res;
      }
      /**
       * allocPageBased tries to allocate the demanded number of bytes on a continues sequence of pages
       * @param bytes number of overall bytes to mark pages for
       * @return pointer to the first page to use, 0 if we were unable to use all the requested pages
       */
      __device__ void* allocPageBased(uint32 bytes)
      {
        //this is rather slow, but we dont expect that to happen often anyway

        //only one thread per warp can acquire the mutex
        void* res = 0;
        for(
#if(__CUDACC_VER_MAJOR__ >= 9)
          unsigned int __mask = __activemask(),
#else
          unsigned int __mask = __ballot(1),
#endif
          __num = __popc(__mask),
          __lanemask = mallocMC::lanemask_lt(),
          __local_id = __popc(__lanemask & __mask),
          __active = 0;
          __active < __num;
          ++__active
        )
          if (__active == __local_id)
            res = allocPageBasedSingle(bytes);
        return res;
      }

      /**
       * deallocPageBased frees the memory placed on a sequence of pages
       * @param mem pointer to the first page
       * @param page the first page
       * @param bytes the number of bytes to be freed
       */
      __device__ void deallocPageBased(void* mem, uint32 page, uint32 bytes)
      {
        uint32 pages = divup(bytes,pagesize);
        for(uint32 p = page; p < page+pages; ++p)
          _page[p].init();
        __threadfence();
        for(uint32 p = page; p < page+pages; ++p)
          atomicCAS((uint32*)&_ptes[p].chunksize, bytes, 0);
        atomicMax((uint32*)&_firstFreePageBased, page+pages-1);
      }


    public:
      /**
       * create allocates the requested number of bytes via the heap. Coalescing has to be done before by another policy.
       * @param bytes number of bytes to allocate
       * @return pointer to the allocated memory
       */
      __device__ void* create(uint32 bytes)
      {
        if(bytes == 0)
          return 0;
        //take care of padding
        //bytes = (bytes + dataAlignment - 1) & ~(dataAlignment-1); // in alignment-policy
        if(bytes < pagesize)
          //chunck based
          return allocChunked(bytes);
        else
          //allocate a range of pages
          return allocPageBased(bytes);
      }

      /**
       * destroy frees the memory regions previously acllocted via create
       * @param mempointer to the memory region to free
       */
      __device__ void destroy(void* mem)
      {
        if(mem == 0)
          return;
        //lets see on which page we are on
        uint32 page = ((char*)mem - (char*)_page)/pagesize;
        uint32 chunksize = _ptes[page].chunksize;

        //is the pointer the beginning of a chunk?
        uint32 inpage_offset = ((char*)mem - _page[page].data);
        uint32 block = inpage_offset/chunksize;
        uint32 inblockoffset = inpage_offset - block*chunksize;
        if(inblockoffset != 0)
        {
          uint32* counter = (uint32*)(_page[page].data + block*chunksize);
          //coalesced mem free
          uint32 old = atomicSub(counter, 1);
          if(old != 1)
            return;
          mem = (void*) counter;
        }

        if(chunksize < pagesize)
          deallocChunked(mem, page, chunksize);
        else
          deallocPageBased(mem, page, chunksize);
      }

      /**
       * init inits the heap data structures
       * the init method must be called before the heap can be used. the method can be called
       * with an arbitrary number of threads, which will increase the inits efficiency
       * @param memory pointer to the memory used for the heap
       * @param memsize size of the memory in bytes
       */
      __device__ void initDeviceFunction(void* memory, size_t memsize)
      {
        uint32 linid = threadIdx.x + blockDim.x*(threadIdx.y + threadIdx.z*blockDim.y);
        uint32 threads = blockDim.x*blockDim.y*blockDim.z;
        uint32 linblockid = blockIdx.x + gridDim.x*(blockIdx.y + blockIdx.z*gridDim.y);
        uint32 blocks =  gridDim.x*gridDim.y*gridDim.z;
        linid = linid + linblockid*threads;

        uint32 numregions = ((unsigned long long)memsize)/( ((unsigned long long)regionsize)*(sizeof(PTE)+pagesize)+sizeof(uint32));
        uint32 numpages = numregions*regionsize;
        //pointer is copied (copy is called page)
        PAGE* page = (PAGE*)(memory);
        //sec check for alignment
        //copy is checked
        //PointerEquivalent alignmentstatus = ((PointerEquivalent)page) & (16 -1);
        //if(alignmentstatus != 0)
        //{
        //  if(linid == 0){
        //    printf("c Before:\n");
        //    printf("c dataAlignment:   %d\n",16);
        //    printf("c Alignmentstatus: %d\n",alignmentstatus);
        //    printf("c size_t memsize   %llu byte\n", memsize);
        //    printf("c void *memory     %p\n", page);
        //  }
        //  //copy is adjusted, potentially pointer to higher address now.
        //  page =(PAGE*)(((PointerEquivalent)page) + 16 - alignmentstatus);
        //  if(linid == 0) printf("c Heap Warning: memory to use not 16 byte aligned...\n");
        //}
        PTE* ptes = (PTE*)(page + numpages);
        uint32* regions = (uint32*)(ptes + numpages);
        //sec check for mem size
        //this check refers to the original memory-pointer, which was not adjusted!
        if( (void*)(regions + numregions) > (((char*)memory) + memsize) )
        {
          --numregions;
          numpages = min(numregions*regionsize,numpages);
          if(linid == 0) printf("c Heap Warning: needed to reduce number of regions to stay within memory limit\n");
        }
        //if(linid == 0) printf("Heap info: wasting %d bytes\n",(((POINTEREQUIVALENT)memory) + memsize) - (POINTEREQUIVALENT)(regions + numregions));

        //if(linid == 0 && alignmentstatus != 0){
        //  printf("c Was shrinked automatically to:\n");
        //  printf("c size_t memsize   %llu byte\n", memsize);
        //  printf("c void *memory     %p\n", page);
        //}
        threads = threads*blocks;

        for(uint32 i = linid; i < numpages; i+= threads)
        {
          ptes[i].init();
          page[i].init();
        }
        for(uint32 i = linid; i < numregions; i+= threads)
          regions[i] = 0;

        if(linid == 0)
        {
          _memsize = memsize;
          _numpages = numpages;
          _ptes = (volatile PTE*)ptes;
          _page = page;
          _regions =  regions;
          _firstfreeblock = 0;
          _pagebasedMutex = 0;
          _firstFreePageBased = numpages-1;

          if( (char*) (_page+numpages) > (char*)(memory) + memsize)
            printf("error in heap alloc: numpages too high\n");
        }

      }

      __device__ bool isOOM(void* p, size_t s){
        // one thread that requested memory returned null
        return  s && (p == NULL);
      }


      template < typename T_DeviceAllocator >
      static void* initHeap( T_DeviceAllocator* heap, void* pool, size_t memsize){
        if( pool == NULL && memsize != 0 )
        {
          throw std::invalid_argument(
            "Scatter policy cannot use NULL for non-empty memory pools. "
            "Maybe you are using an incompatible ReservePoolPolicy or AlignmentPolicy."
          );
        }
        ScatterKernelDetail::initKernel<<<1,256>>>(heap, pool, memsize);
        return heap;
      }

      /** counts how many elements of a size fit inside a given page
       *
       * Examines a (potentially already used) page to find how many elements
       * of size chunksize still fit on the page. This includes hierarchically
       * organized pages and empty pages. The algorithm determines the number
       * of chunks in the page in a manner similar to the allocation algorithm
       * of CreationPolicies::Scatter.
       *
       * @param page the number of the page to examine. The page needs to be
       *        formatted with a chunksize and potentially a hierarchy.
       * @param chunksize the size of element that should be placed inside the
       *        page. This size must be appropriate to the formatting of the
       *        page.
       */
      __device__ unsigned countFreeChunksInPage(uint32 page, uint32 chunksize){
        uint32 filledChunks = _ptes[page].count;
        if(chunksize <= HierarchyThreshold)
        {
          uint32 segmentsize = chunksize*32 + sizeof(uint32); //each segment can hold 32 2nd-level chunks
          uint32 fullsegments = min(32,pagesize / segmentsize); //there might be space for more than 32 segments with 32 2nd-level chunks
          uint32 additional_chunks = calcAdditionalChunks(fullsegments, segmentsize, chunksize);
          uint32 level2Chunks = fullsegments * 32 + additional_chunks;
          return level2Chunks - filledChunks;
        }else{
          uint32 chunksinpage = min(pagesize / chunksize, 32); //without hierarchy, there can not be more than 32 chunks
          return chunksinpage - filledChunks;
        }
      }


      /** counts the number of available slots inside the heap
       *
       * Searches the heap for all possible locations of an element with size
       * slotSize. The used traversal algorithms are similar to the allocation
       * strategy of CreationPolicies::Scatter, to ensure comparable results.
       * There are 3 different algorithms, based on the size of the requested
       * slot: 1 slot spans over multiple pages, 1 slot fits in one chunk
       * within a page, 1 slot fits in a fraction of a chunk.
       *
       * @param slotSize the amount of bytes that a single slot accounts for
       * @param gid the id of the thread. this id does not have to correspond
       *        with threadId.x, but there must be a continous range of ids
       *        beginning from 0.
       * @param stride the stride should be equal to the number of different
       *        gids (and therefore of value max(gid)-1)
       */
      __device__ unsigned getAvailaibleSlotsDeviceFunction(size_t slotSize, int gid, int stride)
      {
        unsigned slotcount = 0;
        if(slotSize < pagesize){ // multiple slots per page
          for(uint32 currentpage = gid; currentpage < _numpages; currentpage += stride){
            uint32 maxchunksize = min(pagesize, wastefactor*(uint32)slotSize);
            uint32 region = currentpage/regionsize;
            uint32 regionfilllevel = _regions[region];

            uint32 chunksize = _ptes[currentpage].chunksize;
            if(chunksize >= slotSize && chunksize <= maxchunksize){ //how many chunks left? (each chunk is big enough)
              slotcount += countFreeChunksInPage(currentpage, chunksize);
            }else if(chunksize == 0){
              chunksize  = max((uint32)slotSize, minChunkSize1); //ensure minimum chunk size
              slotcount += countFreeChunksInPage(currentpage, chunksize); //how many chunks fit in one page?
            }else{
              continue; //the chunks on this page are too small for the request :(
            }
          }
        }else{ // 1 slot needs multiple pages
          if(gid > 0) return 0; //do this serially
          uint32 pagestoalloc = divup((uint32)slotSize, pagesize);
          uint32 freecount = 0;
          for(uint32 currentpage = _numpages; currentpage > 0;){ //this already includes all superblocks
            --currentpage;
            if(_ptes[currentpage].chunksize == 0){
              if(++freecount == pagestoalloc){
                freecount = 0;
                ++slotcount;
              }
            }else{ // the sequence of free pages was interrupted
              freecount = 0;
            }
          }
        }
        return slotcount;
      }


      /** Count, how many elements can be allocated at maximum
       *
       * Takes an input size and determines, how many elements of this size can
       * be allocated with the CreationPolicy Scatter. This will return the
       * maximum number of free slots of the indicated size. It is not
       * guaranteed where these slots are (regarding fragmentation). Therefore,
       * the practically usable number of slots might be smaller. This function
       * is executed in parallel. Speedup can possibly increased by a higher
       * amount ofparallel workers.
       *
       * @param slotSize the size of allocatable elements to count
       * @param obj a reference to the allocator instance (host-side)
       */
    public:
      template<typename T_DeviceAllocator>
      static unsigned getAvailableSlotsHost(size_t const slotSize, T_DeviceAllocator* heap){
        unsigned h_slots = 0;
        unsigned* d_slots;
        cudaMalloc((void**) &d_slots, sizeof(unsigned));
        cudaMemcpy(d_slots, &h_slots, sizeof(unsigned), cudaMemcpyHostToDevice);

        ScatterKernelDetail::getAvailableSlotsKernel<<<64,256>>>(heap, slotSize, d_slots);

        cudaMemcpy(&h_slots, d_slots, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaFree(d_slots);
        return h_slots;
      }


      /** Count, how many elements can be allocated at maximum
       *
       * Takes an input size and determines, how many elements of this size can
       * be allocated with the CreationPolicy Scatter. This will return the
       * maximum number of free slots of the indicated size. It is not
       * guaranteed where these slots are (regarding fragmentation). Therefore,
       * the practically usable number of slots might be smaller. This function
       * is executed separately for each warp and does not cooperate with other
       * warps. Maximum speed is expected if every thread in the warp executes
       * the function.
       * Uses 256 byte of shared memory.
       *
       * @param slotSize the size of allocatable elements to count
       */
      __device__ unsigned getAvailableSlotsAccelerator(size_t slotSize){
        int linearId;
        int wId = warpid_withinblock(); //do not use warpid-function, since this value is not guaranteed to be stable across warp lifetime

#if(__CUDACC_VER_MAJOR__ >= 9)
        uint32 activeThreads  = __popc(__activemask());
#else
        uint32 activeThreads  = __popc(__ballot(true));
#endif
        __shared__ uint32 activePerWarp[MaxThreadsPerBlock::value / WarpSize::value]; //maximum number of warps in a block
        __shared__ unsigned warpResults[MaxThreadsPerBlock::value / WarpSize::value];
        warpResults[wId]   = 0;
        activePerWarp[wId] = 0;

        // the active threads obtain an id from 0 to activeThreads-1
        if(slotSize>0) linearId = atomicAdd(&activePerWarp[wId], 1);
        else return 0;

        //printf("Block %d, id %d: activeThreads=%d linearId=%d\n",blockIdx.x,threadIdx.x,activeThreads,linearId);
        unsigned temp = getAvailaibleSlotsDeviceFunction(slotSize, linearId, activeThreads);
        if(temp) atomicAdd(&warpResults[wId], temp);
        __threadfence_block();
        return warpResults[wId];
      }


      static std::string classname(){
        std::stringstream ss;
        ss << "Scatter[";
        ss << pagesize        << ",";
        ss << accessblocks    << ",";
        ss << regionsize      << ",";
        ss << wastefactor     << ",";
        ss << resetfreedpages << ",";
        ss << hashingK        << ",";
        ss << hashingDistMP   << ",";
        ss << hashingDistWP   << ",";
        ss << hashingDistWPRel<< "]";
        return ss.str();
      }

  };

} //namespace CreationPolicies
} //namespace mallocMC
