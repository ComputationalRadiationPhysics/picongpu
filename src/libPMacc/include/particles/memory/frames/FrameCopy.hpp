/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * libPMacc is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License and the GNU Lesser General Public License 
 * for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * and the GNU Lesser General Public License along with libPMacc. 
 * If not, see <http://www.gnu.org/licenses/>. 
 */ 
 
/* 
 * File:   FrameCopy.hpp
 * Author: widera
 *
 * Created on 14. Januar 2011, 08:05
 */

#ifndef FRAMECOPY_HPP
#define	FRAMECOPY_HPP

#include "types.h"
#include "particles/frame_types.hpp"

namespace PMacc
{
    namespace FrameCopy
    {

        /**
         * FrameCopy specifies copy operations between \{FrameType}s (CoreFrame, BorderFrame, BigFrame).
         * 
         * @tparam DEST_IDENT destination FrameType
         * @tparam SRC_IDENT src FrameType
         */
        template<unsigned DEST_IDENT, unsigned SRC_IDENT>
        class CopyFrame
        {
        public:
            /**
             * Copies src frame to dest frame.
             *
             * Which data (attributes) are copied depends on DEST_IDENT and SRC_IDENT
             * template parameter.
             * 
             * @tparam DEST class of destination frame
             * @tparam SRC class of source frame
             *
             * @param dest destination frame of type DEST
             * @param destId cell id of destination frame
             * @param src source frame of type SRC
             * @param srcId cell id of source frame
             */
            template<class DEST, class SRC>
            HDINLINE void copy(DEST &dest, lcellId_t destId, SRC &src, lcellId_t srcId);
        };

        /**
         * Copies CoreFrame to CoreFrame.
         */
        template<>
        class CopyFrame < CORE_FRAME, CORE_FRAME >
        {
        public:

            template<class DEST, class SRC>
            HDINLINE void copy(DEST &dest, lcellId_t destId, SRC &src, lcellId_t srcId)
            {
                dest.getCellIdx()[destId] = src.getCellIdx()[srcId];
                dest.getPosition()[destId] = src.getPosition()[srcId];
            }

        };

        /**
         * Copies BorderFrame to CoreFrame.
         * superCellId is not set in this copy.
         */
        template<>
        class CopyFrame < CORE_FRAME, BORDER_FRAME >
        {
        public:

            template<class DEST, class SRC>
            HDINLINE void copy(DEST &dest, lcellId_t destId, SRC &src, lcellId_t srcId)
            {
                dest.getCellIdx()[destId] = src.getCellIdx()[srcId];
                dest.getPosition()[destId] = src.getPosition()[srcId];
             //   dest.getMultiMask()[destId] = 1; //!\todo is this a problem to handle masks here?
            }
        };

        /**
         * Copies CoreFrame to BorderFrame.
         * Sets all Masks to initial state.
         */
        template<>
        class CopyFrame < BORDER_FRAME, CORE_FRAME >
        {
        public:

            template<class DEST, class SRC>
            HDINLINE void copy(DEST &dest, lcellId_t destId, SRC &src, lcellId_t srcId)
            {
                dest.getCellIdx()[destId] = src.getCellIdx()[srcId];
                dest.getPosition()[destId] = src.getPosition()[srcId];

            }
        };

        /**
         * Copies BorderFrame to BorderFrame.
         */
        template<>
        class CopyFrame < BORDER_FRAME, BORDER_FRAME >
        {
        public:

            template<class DEST, class SRC>
            HDINLINE void copy(DEST &dest, lcellId_t destId, SRC &src, lcellId_t srcId)
            {
                dest.getCellIdx()[destId] = src.getCellIdx()[srcId];
                dest.getPosition()[destId] = src.getPosition()[srcId];
            }
        };

         /**
         * Copies CoreFrame to BigFrame.
         * superCellId is not set in this copy.
         */
        template<>
        class CopyFrame < BIG_FRAME, CORE_FRAME >
        {
        public:

            template<class DEST, class SRC>
            HDINLINE void copy(DEST &dest, lcellId_t destId, SRC &src, lcellId_t srcId)
            {
                dest.getPosition()[destId] = src.getPosition()[srcId];
            }
        };

    } //namespace FrameCopy

} //namespace PMacc

#endif	/* FRAMECOPY_HPP */

