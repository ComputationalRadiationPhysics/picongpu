#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Axel Huebl, Rene Widera
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

videoName=radiation2800

radiationFolder=../totalRad
folderPNG=../pngElectronsYX
pngPrefix=PngImageElectrons_yx_0.5_

folderRESULT=finale2

deltaT=0.08
nBEGIN=40
nEND=2000
nSTEP=40


folderPDF=tmp/pdf

rm -rf tmp
mkdir tmp
mkdir -p tmp/label
mkdir -p tmp/simPNG
mkdir -p tmp/simPNG2
mkdir -p tmp/pdfPNG
mkdir -p $folderPDF
mkdir $folderRESULT

pwdFolder=`pwd`

cd $folderPDF
#cretae data from radiation file
for((i=$nBEGIN ; i<=$nEND ; i+=$nSTEP )) ; do $pwdFolder/matrix_view.py $pwdFolder/$radiationFolder/RadiationElectrons_$i.dat ; done

cd -


#create timestamp
for((i=$nBEGIN ; i<=$nEND ; i+=$nSTEP )) ; do  x=`echo "$deltaT*$i" | bc -l | sed -r 's/^\./0./g'`; convert -background none -fill white  -size 200x70 -gravity East -pointsize 36 label:"$x fs " tmp/label/`printf "%06i" $i`.png; done

#add timestamp
for((i=$nBEGIN ; i<=$nEND ; i+=$nSTEP )) ; do composite -compose Plus tmp/label/`printf "%06i" $i`.png -gravity NorthEast "$folderPNG/$pngPrefix$i.png"  `printf "tmp/simPNG/%06i" $i`.png; done

#convert pdf to png
for((i=$nBEGIN ; i<=$nEND ; i+=$nSTEP )) ; do x=`printf %05i $i`; convert -depth 8 -resize 1280x720 -quality 100 $folderPDF/spectrum_RadiationElectrons_"$i"_log_GPU_.pdf tmp/pdfPNG/spectrum_"$x".png; done

#create border
for((i=$nBEGIN ; i<=$nEND ; i+=$nSTEP )) ; do x=`printf %06i $i`; convert -depth 8 -bordercolor white -border 192x20 "tmp/simPNG/$x.png" tmp/simPNG2/"$x".png; done

#combine pictures
for((i=$nBEGIN ; i<=$nEND ; i+=$nSTEP )) ; do x=`printf %06i $i`;y=`printf %05i $i`; montage -depth 8 -mode concatenate -tile 1x2 -borderwidth 0 "tmp/simPNG2/$x.png" "tmp/pdfPNG/spectrum_$y.png"  "$folderRESULT/$x.png"; done

cd $folderRESULT
mencoder mf://*.png -mf w=1280:h=720:fps=2:type=png -ovc copy -oac copy -o $pwdFolder/$videoName.avi
cd -
ffmpeg -i $videoName.avi -vcodec libx264 -vpre hq -vpre normal -pass 1 -s 1280x720 -b 2000000 -deinterlace -f mp4 $videoNameD.mp4
ffmpeg -i $videoName.avi -vcodec libx264 -vpre hq -vpre normal -pass 2 -s 1280x720 -b 2000000 -threads 4 -f mp4 $videoName.mp4
#ffmpeg -i $videoName.avi -pix_fmt rgb24 -s 1280x720 $videoName.gif
ffmpeg -i $videoName.avi -y -s 1280x720 -b 2000000 -r 2 -f flv -vcodec flv  $videoName.flv
