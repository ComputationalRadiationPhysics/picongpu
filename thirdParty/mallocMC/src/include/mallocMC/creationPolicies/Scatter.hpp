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

#include "../mallocMC_utils.hpp"
#include "Scatter.hpp"

#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <atomic>
#include <cassert>
#include <cstdint> /* uint32_t */
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

namespace mallocMC
{
    namespace CreationPolicies
    {
        namespace ScatterConf
        {
            struct DefaultScatterConfig
            {
                //! Size in byte of a page.
                static constexpr auto pagesize = 4096;
                /** Size in byte of an access block.
                 *
                 * Scatter alloc will keep allocations within an access block to reduce the translation lookaside
                 * buffer (tlb) pressure. accessblocksize can be used to optimize for the tlb of a device.
                 */
                static constexpr auto accessblocksize = 2u * 1024u * 1024u * 1024u;
                //! Number of pages per region.
                static constexpr auto regionsize = 16;
                //! Factor used to calculate maximal allowed wast depending on the byte.
                static constexpr auto wastefactor = 2;
                /** Defines if a fully freed pages chunk size should be reset.
                 *
                 * true = Chunk size of a page will be reset if free.
                 * false = A page will keep the chunk size selected during the first page usage over
                 *         the full application runtime.
                 */
                static constexpr auto resetfreedpages = false;
            };

            struct DefaultScatterHashingParams
            {
                static constexpr auto hashingK = 38183;
                static constexpr auto hashingDistMP = 17497;
                static constexpr auto hashingDistWP = 1;
                static constexpr auto hashingDistWPRel = 1;
            };
        } // namespace ScatterConf

        /**
         * @brief fast memory allocation based on ScatterAlloc
         *
         * This CreationPolicy implements a fast memory allocator that trades
         * speed for fragmentation of memory. This is based on the memory
         * allocator "ScatterAlloc"
         * (http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6339604),
         * and is extended to report free memory slots of a given size (both on
         * host and accelerator). To work properly, this policy class requires a
         * pre-allocated heap on the accelerator and works only with Nvidia CUDA
         * capable accelerators that have at least compute capability 2.0.
         *
         * @tparam T_Config (optional) configure the heap layout. The
         *        default can be obtained through Scatter<>::HeapProperties
         * @tparam T_Hashing (optional) configure the parameters for
         *        the hashing formula. The default can be obtained through
         *        Scatter<>::HashingProperties
         */
        template<
            class T_Config = ScatterConf::DefaultScatterConfig,
            class T_Hashing = ScatterConf::DefaultScatterHashingParams>
        class Scatter
        {
        public:
            using HeapProperties = T_Config;
            using HashingProperties = T_Hashing;
            struct Properties
                : HeapProperties
                , HashingProperties
            {
            };
            static constexpr auto providesAvailableSlots = true;

        private:
            using uint32 = std::uint32_t;

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
#    define MALLOCMC_CP_SCATTER_PAGESIZE (HeapProperties::pagesize)
#endif
            static constexpr uint32 pagesize = MALLOCMC_CP_SCATTER_PAGESIZE;

#ifndef MALLOCMC_CP_SCATTER_ACCESSBLOCKSIZE
#    define MALLOCMC_CP_SCATTER_ACCESSBLOCKSIZE (HeapProperties::accessblocksize)
#endif
            static constexpr size_t accessblocksize = MALLOCMC_CP_SCATTER_ACCESSBLOCKSIZE;

#ifndef MALLOCMC_CP_SCATTER_REGIONSIZE
#    define MALLOCMC_CP_SCATTER_REGIONSIZE (HeapProperties::regionsize)
#endif
            static constexpr uint32 regionsize = MALLOCMC_CP_SCATTER_REGIONSIZE;

#ifndef MALLOCMC_CP_SCATTER_WASTEFACTOR
#    define MALLOCMC_CP_SCATTER_WASTEFACTOR (HeapProperties::wastefactor)
#endif
            static constexpr uint32 wastefactor = MALLOCMC_CP_SCATTER_WASTEFACTOR;

#ifndef MALLOCMC_CP_SCATTER_RESETFREEDPAGES
#    define MALLOCMC_CP_SCATTER_RESETFREEDPAGES (HeapProperties::resetfreedpages)
#endif
            static constexpr bool resetfreedpages = MALLOCMC_CP_SCATTER_RESETFREEDPAGES;

        public:
            static constexpr uint32 _pagesize = pagesize;
            static constexpr size_t _accessblocksize = accessblocksize;
            static constexpr uint32 _regionsize = regionsize;
            static constexpr uint32 _wastefactor = wastefactor;
            static constexpr bool _resetfreedpages = resetfreedpages;

        private:
#if _DEBUG || ANALYSEHEAP
        public:
#endif
            /* HierarchyThreshold defines the largest chunk size which can be stored in a segment with hierarchy.
             * 32 chunks can be stored without an on page bitmask, therefore a hierarchy is only useful if we store at
             * least 33 chunks. For 33 chunks we need two bitmasks, each 32bit.
             */
            static constexpr uint32 HierarchyThreshold = (pagesize - 2u * sizeof(uint32)) / 33u;
            /* Calculate minimal chunk size which can fill a page, this avoids that small allocations
             * fragment the heap and increases the possibility that a small allocation can reuse an
             * existing chunk.
             * Each page can have 32x32 chunks. To maintain 32 chunks we need 32 bitmask on the page (each 32bit)
             *
             * @note: There is no requirement that minChunksSize is a power of two.
             */
            static constexpr uint32 minChunkSize = (pagesize - 32u * sizeof(uint32)) / (32u * 32u);
            static constexpr uint32 minSegmentSize = 32u * minChunkSize + sizeof(uint32);
            // Number of possible on page masks without taking the limit of 32 masks into account.
            static constexpr uint32 onPageMasks
                = minChunkSize > HierarchyThreshold ? 0u : (pagesize + (minSegmentSize - 1u)) / minSegmentSize;
            // The scatter malloc hierarchy design allows only 32 on page bit masks.
            static constexpr uint32 maxOnPageMasks = std::min(32u, onPageMasks);

#ifndef MALLOCMC_CP_SCATTER_HASHINGK
#    define MALLOCMC_CP_SCATTER_HASHINGK (HashingProperties::hashingK)
#endif
            static constexpr uint32 hashingK = MALLOCMC_CP_SCATTER_HASHINGK;

#ifndef MALLOCMC_CP_SCATTER_HASHINGDISTMP
#    define MALLOCMC_CP_SCATTER_HASHINGDISTMP (HashingProperties::hashingDistMP)
#endif
            static constexpr uint32 hashingDistMP = MALLOCMC_CP_SCATTER_HASHINGDISTMP;

#ifndef MALLOCMC_CP_SCATTER_HASHINGDISTWP
#    define MALLOCMC_CP_SCATTER_HASHINGDISTWP (HashingProperties::hashingDistWP)
#endif
            static constexpr uint32 hashingDistWP = MALLOCMC_CP_SCATTER_HASHINGDISTWP;

#ifndef MALLOCMC_CP_SCATTER_HASHINGDISTWPREL
#    define MALLOCMC_CP_SCATTER_HASHINGDISTWPREL (HashingProperties::hashingDistWPRel)
#endif
            static constexpr uint32 hashingDistWPRel = MALLOCMC_CP_SCATTER_HASHINGDISTWPREL;

            /**
             * Page Table Entry struct
             * The PTE holds basic information about each page
             */
            struct PTE
            {
                uint32 chunksize;
                uint32 count;
                uint32 bitmask;

                ALPAKA_FN_ACC void init()
                {
                    chunksize = 0;
                    count = 0;
                    bitmask = 0;
                }
            };

            /**
             * Page struct
             * The page struct is used to access the data on the page more
             * efficiently and to clear the area on the page, which might hold
             * bitsfields later one
             */
            struct Page
            {
                char data[pagesize];

                /**
                 * The pages init method
                 * This method initializes the region on the page which might
                 * hold bit fields when the page is used for a small chunk size
                 * @param previous_chunksize the chunksize which was uses for
                 * the page before
                 */
                ALPAKA_FN_ACC void init()
                {
                    // clear the entire data which can hold bitfields
                    uint32* write = (uint32*) (data + pagesize - (int) (sizeof(uint32) * maxOnPageMasks));
                    while(write < (uint32*) (data + pagesize))
                        *write++ = 0;
                }
            };

            // the data used by the allocator

            volatile PTE* _ptes;
            volatile uint32* _regions;
            Page* _page;
            size_t _memsize;
            uint32 _numpages;
            uint32 _accessblocks;
            uint32 _pagebasedMutex;
            volatile uint32 _firstFreePageBased;
            volatile uint32 _firstfreeblock;

            /**
             * randInit should create an random offset which can be used
             * as the initial position in a bitfield
             */
            static ALPAKA_FN_ACC inline auto randInit() -> uint32
            {
                // start with the laneid offset
                return laneid();
            }

            /**
             * randInextspot delivers the next free spot in a bitfield
             * it searches for the next unset bit to the left of spot and
             * returns its offset. if there are no unset bits to the left
             * then it wraps around
             * @param bitfield the bitfield to be searched for
             * @param spot the spot from which to search to the left, range [0,spots)
             * @param spots number of bits that can be used
             * @return next free spot in the bitfield
             */
            static ALPAKA_FN_ACC inline auto nextspot(uint32 bitfield, uint32 spot, uint32 spots) -> uint32
            {
                const uint32 low_part = (spot + 1) == sizeof(uint32) * CHAR_BIT ? 0u : (bitfield >> (spot + 1));
                const uint32 high_part = (bitfield << (spots - (spot + 1)));
                const uint32 selection_mask = spots == sizeof(uint32) * CHAR_BIT ? ~0 : ((1 << spots) - 1);
                // wrap around the bitfields from the current spot to the left
                bitfield = (high_part | low_part) & selection_mask;
                // compute the step from the current spot in the bitfield
                const uint32 step = ffs(~bitfield);
                // and return the new spot
                return (spot + step) % spots;
            }

            /**
             * onPageMasksPosition returns a pointer to the beginning of the
             * onpagemasks inside a page.
             * @param page the page that holds the masks
             * @param the number of hierarchical page tables (bitfields) that
             * are used inside this mask.
             * @return pointer to the first address inside the page that holds
             * metadata bitfields.
             */
            ALPAKA_FN_ACC inline auto onPageMasksPosition(uint32 page, uint32 nMasks) -> uint32*
            {
                return (uint32*) (_page[page].data + pagesize - (int) sizeof(uint32) * nMasks);
            }

            /**
             * usespot marks finds one free spot in the bitfield, marks it and
             * returns its offset
             * @param bitfield pointer to the bitfield to use
             * @param spots overall number of spots the bitfield is responsible
             * for
             * @return if there is a free spot it returns the spot'S offset,
             * otherwise -1
             */
            template<typename AlpakaAcc>
            static ALPAKA_FN_ACC inline auto usespot(const AlpakaAcc& acc, uint32* bitfield, uint32 spots) -> int
            {
                // get first spot
                uint32 spot = randInit() % spots;
                for(;;)
                {
                    const uint32 mask = 1 << spot;
                    const uint32 old = alpaka::atomicOp<alpaka::AtomicOr>(acc, bitfield, mask);
                    if((old & mask) == 0)
                        return spot;
                    // note: popc(old) == spots should be sufficient,
                    // but if someone corrupts the memory we end up in an
                    // endless loop in here...
                    if(popc(old) >= spots)
                        return -1;
                    spot = nextspot(old, spot, spots);
                }
            }

            /**
             * calcAdditionalChunks determines the number of chunks that are
             * contained in the last segment of a hierarchical page
             *
             * The additional checks are necessary to ensure correct results for
             * very large pages and small chunksizes
             *
             * @param fullsegments the number of segments that can be completely
             * filled in a page. This may NEVER be bigger than 32!
             * @param segmentsize the number of bytes that are contained in a
             * completely filled segment (32 chunks)
             * @param chunksize the chosen allocation size within the page
             * @return the number of additional chunks that will not fit in one
             * of the fullsegments. For any correct input, this number is
             * smaller than 32
             */
            template<typename AlpakaAcc>
            static ALPAKA_FN_ACC inline auto calcAdditionalChunks(
                const AlpakaAcc& acc,
                uint32 fullsegments,
                uint32 segmentsize,
                uint32 chunksize) -> uint32
            {
                if(fullsegments != 32)
                    return alpaka::math::min(
                        acc,
                        31,
                        alpaka::math::max(
                            acc,
                            0,
                            (int) pagesize - (int) fullsegments * segmentsize - (int) sizeof(uint32))
                            / chunksize);
                else
                    return 0;
            }

            /**
             * addChunkHierarchy finds a free chunk on a page which uses bit
             * fields on the page
             * @param chunksize the chunksize of the page
             * @param fullsegments the number of full segments on the page (a 32
             * bits on the page)
             * @param additional_chunks the number of additional chunks in last
             * segment (less than 32 bits on the page)
             * @param page the page to use
             * @return pointer to a free chunk on the page, 0 if we were unable
             * to obtain a free chunk
             */
            template<typename AlpakaAcc>
            ALPAKA_FN_ACC inline auto addChunkHierarchy(
                const AlpakaAcc& acc,
                uint32 chunksize,
                uint32 fullsegments,
                uint32 additional_chunks,
                uint32 page) -> void*
            {
                const uint32 segments = fullsegments + (additional_chunks > 0 ? 1 : 0);
                uint32 spot = randInit() % segments;
                const uint32 mask = _ptes[page].bitmask;
                if((mask & (1 << spot)) != 0)
                    spot = nextspot(mask, spot, segments);
                const uint32 tries = segments - popc(mask);
                uint32* onpagemasks = onPageMasksPosition(page, segments);
                for(uint32 i = 0; i < tries; ++i)
                {
                    const int hspot = usespot(acc, &onpagemasks[spot], spot < fullsegments ? 32 : additional_chunks);
                    if(hspot != -1)
                        return _page[page].data + (32 * spot + hspot) * chunksize;
                    alpaka::atomicOp<alpaka::AtomicOr>(acc, (uint32*) &_ptes[page].bitmask, 1u << spot);
                    spot = nextspot(mask, spot, segments);
                }
                return 0;
            }

            /**
             * addChunkNoHierarchy finds a free chunk on a page which uses the
             * bit fields of the pte only
             * @param chunksize the chunksize of the page
             * @param page the page to use
             * @param spots the number of chunks which fit on the page
             * @return pointer to a free chunk on the page, 0 if we were unable
             * to obtain a free chunk
             */
            template<typename AlpakaAcc>
            ALPAKA_FN_ACC inline auto addChunkNoHierarchy(
                const AlpakaAcc& acc,
                uint32 chunksize,
                uint32 page,
                uint32 spots) -> void*
            {
                const int spot = usespot(acc, (uint32*) &_ptes[page].bitmask, spots);
                if(spot == -1)
                    return 0; // that should be impossible :)
                return _page[page].data + spot * chunksize;
            }

            /**
             * tryUsePage tries to use the page for the allocation request
             * @param page the page to use
             * @param chunksize the chunksize of the page
             * @return pointer to a free chunk on the page, 0 if we were unable
             * to obtain a free chunk
             */
            template<typename AlpakaAcc>
            ALPAKA_FN_ACC inline auto tryUsePage(const AlpakaAcc& acc, uint32 page, uint32 chunksize) -> void*
            {
                void* chunk_ptr = nullptr;

                // increse the fill level
                const uint32 filllevel = alpaka::atomicOp<alpaka::AtomicAdd>(acc, (uint32*) &(_ptes[page].count), 1u);

                // if resetfreedpages == false we do not need to re-check filllevel or chunksize
                bool tryAllocMem = !resetfreedpages;

                // if _ptes[page].count >= pagesize then page is currently freed by another thread
                if(resetfreedpages && filllevel < pagesize)
                {
                    /* Recheck chunk size (it could be that the page got freed in he meanwhile...)
                     * Use atomic to guarantee that no other thread deleted the page and reinitialized
                     * it with another chunk size.
                     */
                    const uint32 current_chunksize = alpaka::atomicOp<alpaka::AtomicCas>(
                        acc,
                        (uint32*) &_ptes[page].chunksize,
                        chunksize,
                        chunksize);
                    if(current_chunksize == chunksize)
                        tryAllocMem = true;
                }

                if(tryAllocMem)
                {
                    if(chunksize <= HierarchyThreshold)
                    {
                        // more chunks than can be covered by the pte's single
                        // bitfield can be used
                        const uint32 segmentsize = chunksize * 32 + sizeof(uint32);
                        const uint32 fullsegments = alpaka::math::min(acc, 32u, pagesize / segmentsize);
                        const uint32 additional_chunks
                            = calcAdditionalChunks(acc, fullsegments, segmentsize, chunksize);
                        if(filllevel < fullsegments * 32 + additional_chunks)
                            chunk_ptr = addChunkHierarchy(acc, chunksize, fullsegments, additional_chunks, page);
                    }
                    else
                    {
                        const uint32 chunksinpage = alpaka::math::min(acc, pagesize / chunksize, 32u);
                        if(filllevel < chunksinpage)
                            chunk_ptr = addChunkNoHierarchy(acc, chunksize, page, chunksinpage);
                    }
                }

                // this one is full/not useable
                if(chunk_ptr == nullptr)
                    alpaka::atomicOp<alpaka::AtomicSub>(acc, (uint32*) &(_ptes[page].count), 1u);

                return chunk_ptr;
            }

            /**
             * allocChunked tries to allocate the demanded number of bytes on
             * one of the pages
             * @param bytes the number of bytes to allocate, must be <=pagesize
             * @return pointer to a free chunk on a page, 0 if we were unable to
             * obtain a free chunk
             */
            template<typename AlignmentPolicy, typename AlpakaAcc>
            ALPAKA_FN_ACC auto allocChunked(const AlpakaAcc& acc, uint32 bytes) -> void*
            {
                // use the minimal allocation size to increase the hit rate for small allocations.
                const uint32 paddedMinChunkSize = AlignmentPolicy::applyPadding(minChunkSize);
                const uint32 minAllocation = alpaka::math::max(acc, bytes, paddedMinChunkSize);
                const uint32 numpages = _numpages;
                const uint32 pagesperblock = numpages / _accessblocks;
                const uint32 reloff = warpSize * minAllocation / pagesize;
                const uint32 start_page_in_block = (minAllocation * hashingK + hashingDistMP * smid()
                                                    + (hashingDistWP + hashingDistWPRel * reloff) * warpid())
                    % pagesperblock;
                const uint32 maxchunksize = alpaka::math::min(
                    acc,
                    +pagesize,
                    /* this clumping means that allocations of paddedMinChunkSize could have a waste exceeding the
                     * wastefactor
                     */
                    alpaka::math::max(acc, wastefactor * bytes, paddedMinChunkSize));

                /* global page index
                 *   - different for each thread to reduce memory read/write conflicts
                 *   - index calculated by the hash function
                 */
                const uint32 global_start_page = start_page_in_block + _firstfreeblock * pagesperblock;

                uint32 checklevel = regionsize * 3 / 4;
                /* Finding a free segment is using a two step approach.
                 * In both cases each thread will start on a different region and page based on the hash function
                 * result, this scatters the memory access and reduces access conflicts. Both steps will in the worst
                 * case iterate over all heap access blocks and pages.
                 * - step I search for a region which is only filled 3/4
                 *   - if a free segment is found return
                 * - step II goto any region independent of the fill level
                 *   - if a free segment is found return
                 */
                for(uint32 finder = 0; finder < 2; ++finder)
                {
                    uint32 global_page = global_start_page;
                    /* Loop over all pages until we found a free one or arrived to global_start_page again
                     * This and the following loop are done as do-while to potentially save registers by avoiding an
                     * extra loop counter variable
                     */
                    do
                    {
                        const uint32 region = global_page / regionsize;
                        const uint32 regionfilllevel = _regions[region];
                        const uint32 region_offset = region * regionsize;
                        if(regionfilllevel < checklevel)
                        {
                            uint32 page_in_region = global_page;
                            // loop over pages within a region
                            do
                            {
                                const uint32 chunksize = _ptes[page_in_region].chunksize;
                                if(chunksize >= bytes && chunksize <= maxchunksize)
                                {
                                    void* res = tryUsePage(acc, page_in_region, chunksize);
                                    if(res != nullptr)
                                        return res;
                                }
                                else if(chunksize == 0)
                                {
                                    // lets open up a new page
                                    const uint32 beforechunksize = alpaka::atomicOp<alpaka::AtomicCas>(
                                        acc,
                                        (uint32*) &_ptes[page_in_region].chunksize,
                                        0u,
                                        minAllocation);
                                    if(beforechunksize == 0)
                                    {
                                        void* res = tryUsePage(acc, page_in_region, minAllocation);
                                        if(res != nullptr)
                                            return res;
                                    }
                                    else if(beforechunksize >= bytes && beforechunksize <= maxchunksize)
                                    {
                                        // someone else aquired the page,
                                        // but we can also use it
                                        void* res = tryUsePage(acc, page_in_region, beforechunksize);
                                        if(res != nullptr)
                                            return res;
                                    }
                                }
                                page_in_region = region_offset + ((page_in_region + 1) % regionsize);
                            } while(page_in_region != global_page);

                            // could not alloc in region, tell that
                            if(regionfilllevel + 1 <= regionsize)
                                alpaka::atomicOp<alpaka::AtomicCas>(
                                    acc,
                                    (uint32*) (_regions + region),
                                    regionfilllevel,
                                    regionfilllevel + 1);
                        }
                        // goto next region
                        global_page = (global_page + regionsize) % numpages;
                        // check if we jumped into the next access block
                        if(global_page % pagesperblock == 0u)
                        {
                            const uint32 access_block_id = global_page / pagesperblock;
                            // randomize the thread writing the info
                            // Data races are not critical.
                            if(access_block_id > _firstfreeblock)
                                _firstfreeblock = access_block_id;
                        }

                    } while(global_page != global_start_page);

                    // we are really full :/ so lets search every page for a segment!
                    checklevel = regionsize + 1;
                }
                return nullptr;
            }

            /**
             * deallocChunked frees the chunk on the page and updates all data
             * accordingly
             * @param mem pointer to the chunk
             * @param page the page the chunk is on
             * @param chunksize the chunksize used for the page
             */
            template<typename AlpakaAcc>
            ALPAKA_FN_ACC void deallocChunked(const AlpakaAcc& acc, void* mem, uint32 page, uint32 chunksize)
            {
                const auto inpage_offset = static_cast<uint32>((char*) mem - _page[page].data);
                if(chunksize <= HierarchyThreshold)
                {
                    // one more level in hierarchy
                    const uint32 segmentsize = chunksize * 32 + sizeof(uint32);
                    const uint32 fullsegments = alpaka::math::min(acc, 32u, pagesize / segmentsize);
                    const uint32 additional_chunks = calcAdditionalChunks(acc, fullsegments, segmentsize, chunksize);
                    const uint32 segment = inpage_offset / (chunksize * 32);
                    const uint32 withinsegment = (inpage_offset - segment * (chunksize * 32)) / chunksize;
                    // mark it as free
                    const uint32 nMasks = fullsegments + (additional_chunks > 0 ? 1 : 0);
                    uint32* onpagemasks = onPageMasksPosition(page, nMasks);
                    uint32 old
                        = alpaka::atomicOp<alpaka::AtomicAnd>(acc, &onpagemasks[segment], ~(1u << withinsegment));

                    // always do this, since it might fail due to a
                    // race-condition with addChunkHierarchy
                    alpaka::atomicOp<alpaka::AtomicAnd>(acc, (uint32*) &_ptes[page].bitmask, ~(1u << segment));
                }
                else
                {
                    const uint32 segment = inpage_offset / chunksize;
                    alpaka::atomicOp<alpaka::AtomicAnd>(acc, (uint32*) &_ptes[page].bitmask, ~(1u << segment));
                }
                // reduce filllevel as free
                const uint32 oldfilllevel = alpaka::atomicOp<alpaka::AtomicSub>(acc, (uint32*) &_ptes[page].count, 1u);

                if(resetfreedpages)
                {
                    if(oldfilllevel == 1)
                    {
                        // this page now got free!
                        // -> try lock it
                        const uint32 old
                            = alpaka::atomicOp<alpaka::AtomicCas>(acc, (uint32*) &_ptes[page].count, 0u, +pagesize);
                        if(old == 0)
                        {
                            // clean the bits for the hierarchy
                            _page[page].init();
                            // remove chunk information
                            alpaka::atomicOp<alpaka::AtomicCas>(acc, (uint32*) &_ptes[page].chunksize, chunksize, 0u);

                            threadfenceDevice(acc);

                            // unlock it
                            alpaka::atomicOp<alpaka::AtomicSub>(acc, (uint32*) &_ptes[page].count, +pagesize);
                        }
                    }
                }

                // meta information counters ... should not be changed by too
                // many threads, so..
                if(oldfilllevel == pagesize / 2 / chunksize)
                {
                    const uint32 region = page / regionsize;
                    alpaka::atomicOp<alpaka::AtomicExch>(acc, (uint32*) (_regions + region), 0u);
                    const uint32 block = region * regionsize * _accessblocks / _numpages;
                    if(warpid() + laneid() == 0)
                        alpaka::atomicOp<alpaka::AtomicMin>(acc, (uint32*) &_firstfreeblock, block);
                }
            }

            /**
             * markpages markes a fixed number of pages as used
             * @param startpage first page to mark
             * @param pages number of pages to mark
             * @param bytes number of overall bytes to mark pages for
             * @return true on success, false if one of the pages is not free
             */
            template<typename AlpakaAcc>
            ALPAKA_FN_ACC auto markpages(const AlpakaAcc& acc, uint32 startpage, uint32 pages, uint32 bytes) -> bool
            {
                int abord = -1;
                for(uint32 trypage = startpage; trypage < startpage + pages; ++trypage)
                {
                    const uint32 old
                        = alpaka::atomicOp<alpaka::AtomicCas>(acc, (uint32*) &_ptes[trypage].chunksize, 0u, bytes);
                    if(old != 0)
                    {
                        abord = trypage;
                        break;
                    }
                }
                if(abord == -1)
                    return true;
                for(uint32 trypage = startpage; trypage < abord; ++trypage)
                    alpaka::atomicOp<alpaka::AtomicCas>(acc, (uint32*) &_ptes[trypage].chunksize, bytes, 0u);
                return false;
            }

            /**
             * allocPageBasedSingleRegion tries to allocate the demanded number
             * of bytes on a continues sequence of pages
             * @param startpage first page to be used
             * @param endpage last page to be used
             * @param bytes number of overall bytes to mark pages for
             * @return pointer to the first page to use, 0 if we were unable to
             * use all the requested pages
             */
            template<typename AlpakaAcc>
            ALPAKA_FN_ACC auto allocPageBasedSingleRegion(
                const AlpakaAcc& acc,
                uint32 startpage,
                uint32 endpage,
                uint32 bytes) -> void*
            {
                const uint32 pagestoalloc = divup(bytes, pagesize);
                uint32 freecount = 0;
                bool left_free = false;
                for(uint32 search_page = startpage + 1; search_page > endpage;)
                {
                    --search_page;
                    if(_ptes[search_page].chunksize == 0)
                    {
                        if(++freecount == pagestoalloc)
                        {
                            // try filling it up
                            if(markpages(acc, search_page, pagestoalloc, bytes))
                            {
                                // mark that we filled up everything up to here
                                if(!left_free)
                                    alpaka::atomicOp<alpaka::AtomicCas>(
                                        acc,
                                        (uint32*) &_firstFreePageBased,
                                        startpage,
                                        search_page - 1);
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
             * allocPageBasedSingle tries to allocate the demanded number of
             * bytes on a continues sequence of pages
             * @param bytes number of overall bytes to mark pages for
             * @return pointer to the first page to use, 0 if we were unable to
             * use all the requested pages
             * @pre only a single thread of a warp is allowed to call the
             * function concurrently
             */
            template<typename AlpakaAcc>
            ALPAKA_FN_ACC auto allocPageBasedSingle(const AlpakaAcc& acc, uint32 bytes) -> void*
            {
                // acquire mutex
                while(alpaka::atomicOp<alpaka::AtomicExch>(acc, &_pagebasedMutex, 1u) != 0)
                    ;
                // search for free spot from the back
                const uint32 spage = _firstFreePageBased;
                void* res = allocPageBasedSingleRegion(acc, spage, 0, bytes);
                if(res == 0)
                    // also check the rest of the pages
                    res = allocPageBasedSingleRegion(acc, _numpages, spage, bytes);

                // free mutex
                alpaka::atomicOp<alpaka::AtomicExch>(acc, &_pagebasedMutex, 0u);
                return res;
            }
            /**
             * allocPageBased tries to allocate the demanded number of bytes on
             * a continues sequence of pages
             * @param bytes number of overall bytes to mark pages for
             * @return pointer to the first page to use, 0 if we were unable to
             * use all the requested pages
             */
            template<typename AlpakaAcc>
            ALPAKA_FN_ACC auto allocPageBased(const AlpakaAcc& acc, uint32 bytes) -> void*
            {
                // this is rather slow, but we dont expect that to happen often
                // anyway

                // only one thread per warp can acquire the mutex
                void* res = 0;
                // based on the alpaka backend the lanemask type can be 64bit
                const auto mask = activemask();
                const uint32_t num = popc(mask);
                // based on the alpaka backend the lanemask type can be 64bit
                const auto lanemask = lanemask_lt();
                const uint32_t local_id = popc(lanemask & mask);
                for(unsigned int active = 0; active < num; ++active)
                    if(active == local_id)
                        res = allocPageBasedSingle(acc, bytes);
                return res;
            }

            /**
             * deallocPageBased frees the memory placed on a sequence of pages
             * @param mem pointer to the first page
             * @param page the first page
             * @param bytes the number of bytes to be freed
             */
            template<typename AlpakaAcc>
            ALPAKA_FN_ACC void deallocPageBased(const AlpakaAcc& acc, void* mem, uint32 page, uint32 bytes)
            {
                const uint32 pages = divup(bytes, pagesize);
                for(uint32 p = page; p < page + pages; ++p)
                    _page[p].init();

                threadfenceDevice(acc);

                for(uint32 p = page; p < page + pages; ++p)
                    alpaka::atomicOp<alpaka::AtomicCas>(acc, (uint32*) &_ptes[p].chunksize, bytes, 0u);
                alpaka::atomicOp<alpaka::AtomicMax>(acc, (uint32*) &_firstFreePageBased, page + pages - 1);
            }

        public:
            /**
             * create allocates the requested number of bytes via the heap.
             * Coalescing has to be done before by another policy.
             * @param bytes number of bytes to allocate
             * @return pointer to the allocated memory
             */
            template<typename AlignmentPolicy, typename AlpakaAcc>
            ALPAKA_FN_ACC auto create(const AlpakaAcc& acc, uint32 bytes) -> void*
            {
                if(bytes == 0)
                    return 0;
                /* Take care of padding
                 * bytes = (bytes + dataAlignment - 1) & ~(dataAlignment-1);
                 * in alignment-policy.
                 * bytes == pagesize must be handled by allocChunked() else maxchunksize calculation based
                 * on the waste factor is colliding with the allocation schema in allocPageBased().
                 */
                if(bytes <= pagesize)
                    // chunck based
                    return allocChunked<AlignmentPolicy>(acc, bytes);
                else
                    // allocate a range of pages
                    return allocPageBased(acc, bytes);
            }

            /**
             * destroy frees the memory regions previously acllocted via create
             * @param mempointer to the memory region to free
             */
            template<typename AlpakaAcc>
            ALPAKA_FN_ACC void destroy(const AlpakaAcc& acc, void* mem)
            {
                if(mem == 0)
                    return;
                // lets see on which page we are on
                const auto page = static_cast<uint32>(((char*) mem - (char*) _page) / pagesize);
                const uint32 chunksize = _ptes[page].chunksize;

                // is the pointer the beginning of a chunk?
                const auto inpage_offset = static_cast<uint32>((char*) mem - _page[page].data);
                const uint32 block = inpage_offset / chunksize;
                const uint32 inblockoffset = inpage_offset - block * chunksize;
                if(inblockoffset != 0)
                {
                    uint32* counter = (uint32*) (_page[page].data + block * chunksize);
                    // coalesced mem free

                    const uint32 old = alpaka::atomicOp<alpaka::AtomicSub>(acc, counter, 1u);
                    if(old != 1)
                        return;
                    mem = (void*) counter;
                }

                if(chunksize < pagesize)
                    deallocChunked(acc, mem, page, chunksize);
                else
                    deallocPageBased(acc, mem, page, chunksize);
            }

            /**
             * init inits the heap data structures
             * the init method must be called before the heap can be used. the
             * method can be called with an arbitrary number of threads, which
             * will increase the inits efficiency
             * @param memory pointer to the memory used for the heap
             * @param memsize size of the memory in bytes
             */
            template<typename AlpakaAcc>
            ALPAKA_FN_ACC void initDeviceFunction(const AlpakaAcc& acc, void* memory, size_t memsize)
            {
                const auto linid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc).sum();
                const auto totalThreads = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc).prod();

                uint32 numregions = ((unsigned long long) memsize)
                    / (((unsigned long long) regionsize) * (sizeof(PTE) + pagesize) + sizeof(uint32));

                uint32 numpages = numregions * regionsize;
                // pointer is copied (copy is called page)
                Page* page = (Page*) memory;
                // sec check for alignment
                // copy is checked
                // PointerEquivalent alignmentstatus = ((PointerEquivalent)page)
                // & (16 -1); if(alignmentstatus != 0)
                //{
                //  if(linid == 0){
                //    printf("c Before:\n");
                //    printf("c dataAlignment:   %d\n",16);
                //    printf("c Alignmentstatus: %d\n",alignmentstatus);
                //    printf("c size_t memsize   %llu byte\n", memsize);
                //    printf("c void *memory     %p\n", page);
                //  }
                //  //copy is adjusted, potentially pointer to higher address
                //  now. page =(Page*)(((PointerEquivalent)page) + 16 -
                //  alignmentstatus); if(linid == 0) printf("c Heap Warning:
                //  memory to use not 16 byte aligned...\n");
                //}

                // We have to calculate these values here, before using them for other things.
                // First calculate how many blocks of the given size fit our memory pages in principle.
                // However, we do not have to use the exact requested block size.
                // So we redistribute actual memory between the chosen number of blocks
                // and ensure that all blocks have the same number of regions.
                const auto memorysize = static_cast<size_t>(numpages) * pagesize;
                const auto numblocks = memorysize / accessblocksize;
                const auto memoryperblock = memorysize / numblocks;
                const auto pagesperblock = memoryperblock / pagesize;
                const auto regionsperblock = pagesperblock / regionsize;
                numregions = numblocks * regionsperblock;
                numpages = numregions * regionsize;

                PTE* ptes = (PTE*) (page + numpages);
                uint32* regions = (uint32*) (ptes + numpages);
                // sec check for mem size
                // this check refers to the original memory-pointer, which was
                // not adjusted!
                if((char*) (regions + numregions) > (((char*) memory) + memsize))
                {
                    --numregions;
                    numpages = alpaka::math::min(acc, numregions * regionsize, numpages);
                    if(linid == 0)
                        printf("c Heap Warning: needed to reduce number of "
                               "regions to stay within memory limit\n");
                }
                // Recalculate since numpages could have changed
                ptes = (PTE*) (page + numpages);
                regions = (uint32*) (ptes + numpages);

                // if(linid == 0) printf("Heap info: wasting %d
                // bytes\n",(((POINTEREQUIVALENT)memory) + memsize) -
                // (POINTEREQUIVALENT)(regions + numregions));

                // if(linid == 0 && alignmentstatus != 0){
                //  printf("c Was shrinked automatically to:\n");
                //  printf("c size_t memsize   %llu byte\n", memsize);
                //  printf("c void *memory     %p\n", page);
                //}

                for(uint32 i = linid; i < numpages; i += totalThreads)
                {
                    ptes[i].init();
                    page[i].init();
                }
                for(uint32 i = linid; i < numregions; i += totalThreads)
                    regions[i] = 0;

                if(linid == 0)
                {
                    _memsize = memsize;
                    _numpages = numpages;
                    _accessblocks = numblocks;
                    _ptes = (volatile PTE*) ptes;
                    _page = page;
                    _regions = regions;
                    _firstfreeblock = 0;
                    _pagebasedMutex = 0;
                    _firstFreePageBased = numpages - 1;

                    if((char*) &_page[numpages] > (char*) memory + memsize)
                        printf("error in heap alloc: numpages too high\n");
                }
            }

            static ALPAKA_FN_ACC auto isOOM(void* p, size_t s) -> bool
            {
                // one thread that requested memory returned null
                return s && (p == nullptr);
            }

            template<typename AlpakaAcc, typename AlpakaDevice, typename AlpakaQueue, typename T_DeviceAllocator>
            static void initHeap(
                AlpakaDevice& dev,
                AlpakaQueue& queue,
                T_DeviceAllocator* heap,
                void* pool,
                size_t memsize)
            {
                if(pool == nullptr && memsize != 0)
                {
                    throw std::invalid_argument("Scatter policy cannot use nullptr for non-empty "
                                                "memory pools. "
                                                "Maybe you are using an incompatible ReservePoolPolicy "
                                                "or AlignmentPolicy.");
                }
                auto initKernel = [] ALPAKA_FN_ACC(
                                      const AlpakaAcc& m_acc,
                                      T_DeviceAllocator* m_heap,
                                      void* m_heapmem,
                                      size_t m_memsize) {
                    m_heap->pool = m_heapmem;
                    m_heap->initDeviceFunction(m_acc, m_heapmem, m_memsize);
                };
                using Dim = typename alpaka::traits::DimType<AlpakaAcc>::type;
                using Idx = typename alpaka::traits::IdxType<AlpakaAcc>::type;
                using VecType = alpaka::Vec<Dim, Idx>;

                auto threadsPerBlock = VecType::ones();

                auto const devProps = alpaka::getAccDevProps<AlpakaAcc>(dev);

                threadsPerBlock[Dim::value - 1]
                    = std::min(static_cast<size_t>(256u), static_cast<size_t>(devProps.m_blockThreadCountMax));

                const auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{
                    VecType::ones(),
                    threadsPerBlock,
                    VecType::ones()}; // Dim may be any dimension, but workDiv is 1D
                alpaka::enqueue(queue, alpaka::createTaskKernel<AlpakaAcc>(workDiv, initKernel, heap, pool, memsize));
            }

            /** counts how many elements of a size fit inside a given page
             *
             * Examines a (potentially already used) page to find how many
             * elements of size chunksize still fit on the page. This includes
             * hierarchically organized pages and empty pages. The algorithm
             * determines the number of chunks in the page in a manner similar
             * to the allocation algorithm of CreationPolicies::Scatter.
             *
             * @param page the number of the page to examine. The page needs to
             * be formatted with a chunksize and potentially a hierarchy.
             * @param chunksize the size of element that should be placed inside
             * the page. This size must be appropriate to the formatting of the
             *        page.
             */
            template<typename AlpakaAcc>
            ALPAKA_FN_ACC auto countFreeChunksInPage(const AlpakaAcc& acc, uint32 page, uint32 chunksize) -> unsigned
            {
                const uint32 filledChunks = _ptes[page].count;
                if(chunksize <= HierarchyThreshold)
                {
                    const uint32 segmentsize = chunksize * 32 + sizeof(uint32); // each segment can hold 32 2nd-level
                                                                                // chunks
                    const uint32 fullsegments = alpaka::math::min(
                        acc,
                        32u,
                        pagesize / segmentsize); // there might be space for
                                                 // more than 32 segments
                                                 // with 32 2nd-level chunks
                    const uint32 additional_chunks = calcAdditionalChunks(acc, fullsegments, segmentsize, chunksize);
                    const uint32 level2Chunks = fullsegments * 32 + additional_chunks;
                    return level2Chunks - filledChunks;
                }
                else
                {
                    const uint32 chunksinpage = alpaka::math::min(
                        acc,
                        pagesize / chunksize,
                        32u); // without hierarchy, there can not be more than
                              // 32 chunks
                    return chunksinpage - filledChunks;
                }
            }

            /** counts the number of available slots inside the heap
             *
             * Searches the heap for all possible locations of an element with
             * size slotSize. The used traversal algorithms are similar to the
             * allocation strategy of CreationPolicies::Scatter, to ensure
             * comparable results. There are 3 different algorithms, based on
             * the size of the requested slot: 1 slot spans over multiple pages,
             * 1 slot fits in one chunk within a page, 1 slot fits in a fraction
             * of a chunk.
             *
             * @param slotSize the amount of bytes that a single slot accounts
             * for
             * @param gid the id of the thread. this id does not have to
             * correspond with threadId.x, but there must be a continous range
             * @param stride the stride should be equal to the number of
             * different gids (and therefore of value max(gid)-1)
             */
            template<typename AlignmentPolicy, typename AlpakaAcc>
            ALPAKA_FN_ACC auto getAvailaibleSlotsDeviceFunction(
                const AlpakaAcc& acc,
                size_t slotSize,
                uint32 gid,
                uint32 stride) -> unsigned
            {
                unsigned slotcount = 0;
                if(slotSize < pagesize)
                { // multiple slots per page
                    for(uint32 currentpage = gid; currentpage < _numpages; currentpage += stride)
                    {
                        const uint32 maxchunksize = alpaka::math::min(acc, +pagesize, wastefactor * (uint32) slotSize);

                        uint32 chunksize = _ptes[currentpage].chunksize;
                        if(chunksize >= slotSize && chunksize <= maxchunksize)
                        { // how many chunks left? (each chunk is big enough)
                            slotcount += countFreeChunksInPage(acc, currentpage, chunksize);
                        }
                        else if(chunksize == 0)
                        {
                            chunksize = alpaka::math::max(
                                acc,
                                (uint32) slotSize,
                                AlignmentPolicy::applyPadding(minChunkSize)); // ensure minimum chunk size
                            slotcount += countFreeChunksInPage(
                                acc,
                                currentpage,
                                chunksize); // how many chunks fit in one page?
                        }
                        else
                        {
                            continue; // the chunks on this page are too small
                                      // for the request :(
                        }
                    }
                }
                else
                { // 1 slot needs multiple pages
                    if(gid > 0)
                        return 0; // do this serially
                    const uint32 pagestoalloc = divup((uint32) slotSize, pagesize);
                    uint32 freecount = 0;
                    for(uint32 currentpage = _numpages; currentpage > 0;)
                    { // this already includes all superblocks
                        --currentpage;
                        if(_ptes[currentpage].chunksize == 0)
                        {
                            if(++freecount == pagestoalloc)
                            {
                                freecount = 0;
                                ++slotcount;
                            }
                        }
                        else
                        { // the sequence of free pages was interrupted
                            freecount = 0;
                        }
                    }
                }
                return slotcount;
            }

            /** Count, how many elements can be allocated at maximum
             *
             * Takes an input size and determines, how many elements of this
             * size can be allocated with the CreationPolicy Scatter. This will
             * return the maximum number of free slots of the indicated size. It
             * is not guaranteed where these slots are (regarding
             * fragmentation). Therefore, the practically usable number of slots
             * might be smaller. This function is executed in parallel. Speedup
             * can possibly increased by a higher amount ofparallel workers.
             *
             * @param slotSize the size of allocatable elements to count
             * @param obj a reference to the allocator instance (host-side)
             */
        public:
            template<typename AlpakaAcc, typename AlpakaDevice, typename AlpakaQueue, typename T_DeviceAllocator>
            static auto getAvailableSlotsHost(
                AlpakaDevice& dev,
                AlpakaQueue& queue,
                size_t const slotSize,
                T_DeviceAllocator* heap) -> unsigned
            {
                auto d_slots = alpaka::allocBuf<unsigned, int>(dev, 1);
                alpaka::memset(queue, d_slots, 0, 1);

                auto getAvailableSlotsKernel
                    = [] ALPAKA_FN_ACC(const AlpakaAcc& acc, T_DeviceAllocator* heap, size_t slotSize, unsigned* slots)
                    -> void {
                    const auto gid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc).sum();

                    const auto nWorker = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc).prod();
                    const unsigned temp
                        = heap->template getAvailaibleSlotsDeviceFunction<typename T_DeviceAllocator::AlignmentPolicy>(
                            acc,
                            slotSize,
                            gid,
                            nWorker);
                    if(temp)
                        alpaka::atomicOp<alpaka::AtomicAdd>(acc, slots, temp);
                };

                using Dim = typename alpaka::traits::DimType<AlpakaAcc>::type;
                using Idx = typename alpaka::traits::IdxType<AlpakaAcc>::type;

                using VecType = alpaka::Vec<Dim, Idx>;

                auto numBlocks = VecType::ones();
                numBlocks[Dim::value - 1] = 64u;
                auto threadsPerBlock = VecType::ones();

                auto const devProps = alpaka::getAccDevProps<AlpakaAcc>(dev);

                threadsPerBlock[Dim::value - 1]
                    = std::min(static_cast<size_t>(256u), static_cast<size_t>(devProps.m_blockThreadCountMax));

                const auto workDiv = alpaka::WorkDivMembers<Dim, Idx>{
                    numBlocks,
                    threadsPerBlock,
                    VecType::ones()}; // Dim may be any dimension, but workDiv is 1D

                alpaka::enqueue(
                    queue,
                    alpaka::createTaskKernel<AlpakaAcc>(
                        workDiv,
                        getAvailableSlotsKernel,
                        heap,
                        slotSize,
                        alpaka::getPtrNative(d_slots)));

                const auto hostDev = alpaka::getDevByIdx<alpaka::Pltf<alpaka::DevCpu>>(0);
                auto h_slots = alpaka::allocBuf<unsigned, int>(hostDev, 1);
                alpaka::memcpy(queue, h_slots, d_slots, 1);
                alpaka::wait(queue);

                return *alpaka::getPtrNative(h_slots);
            }

            /** Count, how many elements can be allocated at maximum
             *
             * Takes an input size and determines, how many elements of this
             * size can be allocated with the CreationPolicy Scatter. This will
             * return the maximum number of free slots of the indicated size. It
             * is not guaranteed where these slots are (regarding
             * fragmentation). Therefore, the practically usable number of slots
             * might be smaller. This function is executed separately for each
             * warp and does not cooperate with other warps. Maximum speed is
             * expected if every thread in the warp executes the function. Uses
             * 256 byte of shared memory.
             *
             * @param slotSize the size of allocatable elements to count
             */
            template<typename AlignmentPolicy, typename AlpakaAcc>
            ALPAKA_FN_ACC auto getAvailableSlotsAccelerator(const AlpakaAcc& acc, size_t slotSize) -> unsigned
            {
                const int wId = warpid_withinblock(acc); // do not use warpid-function, since
                                                         // this value is not guaranteed to
                                                         // be stable across warp lifetime

                const uint32 activeThreads = popc(activemask());

                auto& activePerWarp = alpaka::declareSharedVar<
                    std::uint32_t[maxThreadsPerBlock / warpSize],
                    __COUNTER__>(acc); // maximum number of warps in a block

                auto& warpResults
                    = alpaka::declareSharedVar<unsigned[maxThreadsPerBlock / warpSize], __COUNTER__>(acc);

                warpResults[wId] = 0;
                activePerWarp[wId] = 0;

                // wait that all shared memory is initialized
                alpaka::syncBlockThreads(acc);

                // the active threads obtain an id from 0 to activeThreads-1
                if(slotSize == 0)
                    return 0;
                const auto linearId = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &activePerWarp[wId], 1u);

                // printf("Block %d, id %d: activeThreads=%d
                // linearId=%d\n",blockIdx.x,threadIdx.x,activeThreads,linearId);
                const unsigned temp = this->template getAvailaibleSlotsDeviceFunction<AlignmentPolicy>(
                    acc,
                    slotSize,
                    linearId,
                    activeThreads);
                if(temp)
                    alpaka::atomicOp<alpaka::AtomicAdd>(acc, &warpResults[wId], temp);

                alpaka::syncBlockThreads(acc);
                threadfenceBlock(acc);

                return warpResults[wId];
            }

            static auto classname() -> std::string
            {
                std::stringstream ss;
                ss << "Scatter[";
                ss << "pagesize=" << pagesize << ",";
                ss << "accessblocksize=" << accessblocksize << ",";
                ss << "regionsize=" << regionsize << ",";
                ss << "wastefactor=" << wastefactor << ",";
                ss << "resetfreedpages=" << resetfreedpages << ",";
                ss << "minChunkSize=" << minChunkSize << ",";
                ss << "HierarchyThreshold=" << HierarchyThreshold << ",";
                ss << "hashingK=" << hashingK << ",";
                ss << "hashingDistMP=" << hashingDistMP << ",";
                ss << "hashingDistWP=" << hashingDistWP << ",";
                ss << "hashingDistWPRel=" << hashingDistWPRel << "]";
                return ss.str();
            }
        };

    } // namespace CreationPolicies
} // namespace mallocMC
