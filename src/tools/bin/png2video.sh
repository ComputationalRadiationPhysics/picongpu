#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Rene Widera
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

videoName=radiation2800

radiationFolder=../totalRad
folderPNG=../pngElectronsYX
pngPrefix=PngImageElectrons_yx_0.5_

folderRESULT=finale2

deltaT=0.16
nBEGIN=20
nEND=2800
nSTEP=20


folderPDF=tmp/pdf



pwdFolder=`pwd`


cd $folderRESULT
mencoder mf://*.png -mf w=1280:h=720:fps=2:type=png -ovc copy -oac copy -o $pwdFolder/$videoName.avi
cd -
ffmpeg -i $videoName.avi -vcodec libx264 -vpre hq -vpre normal -pass 1 -s 1280x1192 -b 2000000 -deinterlace -f mp4 "$videoName.mp4"
ffmpeg -i $videoName.avi -vcodec libx264 -vpre hq -vpre normal -pass 2 -s 1280x1192 -b 2000000 -threads 4 -f mp4 "$videoName.mp4"
#ffmpeg -i $videoName.avi -pix_fmt rgb24 -s 1280x1192 "$videoName.gif"
ffmpeg -i $videoName.avi -y -s 1280x1192 -b 2000000 -r 2 -f flv -vcodec flv  "$videoName.flv"
