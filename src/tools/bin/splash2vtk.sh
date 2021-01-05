#!/usr/bin/env bash
#
# Copyright 2013-2021 Axel Huebl
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

filename="$1"
step=1000
dataNames="fields_E_y"
dataEl=`echo "$dataNames" | wc -w`

# dimensions & meta data
n=`splash2txt -l $filename 2>&1 | grep "domain"`
nx=`echo $n | awk -F, '{print $4}' | tr -cd '0-9\012'`
ny=`echo $n | awk -F, '{print $5}' | tr -cd '0-9\012'`
nz=`echo $n | awk -F, '{print $6}' | tr -cd '0-9\012'`

# cell spacing
dx=1
dy=1
dz=1

# slice width
sw=20

# little endian = 1, else 0
littleEndian=`echo I | tr -d [:space:] | od -to2 | head -n1 | awk '{print $2}' | cut -c6`
byteOrder=""
if [ $littleEndian -eq 1 ]; then
  byteOrder="LittleEndian"
else
  byteOrder="BigEndian"
fi

#for ty in `seq 100 200`
#for tz in `seq 1 $nz`
z=1
while [ $z -le $nz ]
do
  # max of this slice
  zm=$((z+10))
  if [ $zm -gt $nz ]; then
    zm=$nz
  fi

  # VTK Header for GRID DATA slab
  rm -rf slab_$zm.vti
  cat >> slab_$zm.vti <<EOF
<?xml version="1.0" ?>
<VTKFile type="ImageData" version="0.1" byte_order="$byteOrder">
 <ImageData WholeExtent="1 $nx 1 $ny $z $zm" Origin="0 0 0" Spacing="$dx $dy $dz">
EOF

  #while [ $z -le $zm ]
  #do
    cat >> slab_$zm.vti <<EOF
  <Piece Extent="1 $nx 1 $ny $z $zm">
   <PointData>
    <DataArray Name="fieldData" format="ascii" type="Float64" NumberOfComponents="$dataEl">
EOF
    #      replace every 3rd "," with newline                      replace , with space   replace dbl newlines
    #sed -e 's/\(\([^,]*,\)\{2\}[^,]*\),/\1\n/g' slab_pure_$y.dat | sed -e 's/,/ /g' | sed '/^$/d' >> slab_$y.vti
    #      replace , with a space         replace every newline
    #sed -e 's/,/ /g' slab_pure_$z.dat | sed ':a;N;$!ba;s/\n/ /g' | sed '/^$/d' >> slab_$z.vti

    while [ $z -le $zm ]
    do
      # read slab
      #splash2txt --slice xy -s $step -d $dataNames --offset $((z-1)) --delimiter " " -o slab_pure_$zm.dat $filename
      splash2txt --slice xy -s $step -d $dataNames --offset $((z-1)) --delimiter " " --input-file $filename | tr -d '\n' >> slab_$zm.vti
      #    replace every newline
      #sed -e ':a;N;$!ba;s/\n/ /g' slab_pure_$zm.dat >> slab_$zm.vti
      #rm -rf slab_pure_$zm.dat

      z=$((z+1))
    done

    cat >> slab_$zm.vti <<EOF
    </DataArray>
   </PointData>
   <CellData>
   </CellData>
  </Piece>
EOF
  #done

  cat >> slab_$zm.vti <<EOF
 </ImageData>
</VTKFile>
EOF

  # <DataArray type="Float64" Name="fieldData" format="ascii" >
  # <ImageData WholeExtent="0 $nx 0 $ny 0 $nz" Origin="0 0 0" Spacing="$dx $dy $dz">
  #  <Piece Extent="0 $nx 0 1 0 $nz">

  # decrease $x by 1 for pvti overlap
  if [ $zm -lt $nz ]; then
    z=$((zm))
  else
    z=$((nz+1))
  fi

  echo "z($z) zm($zm)"

done

#fi

# VTK Master File ##############################################################
rm -rf pslab.pvti

cat >> pslab.pvti <<EOF
<?xml version="1.0" ?>
<VTKFile type="PImageData" version="0.1" byte_order="$byteOrder">
 <PImageData WholeExtent="1 $nx 1 $ny 1 $nz" Origin="0 0 0" Spacing="$dx $dy $dz" GhostLevel="0">
  <PPointData>
   <PDataArray Name="fieldData" type="Float64" NumberOfComponents="$dataEl"/>
  </PPointData>
  <PCellData>
  </PCellData>
EOF

#for ty in `seq 100 200`
#for tz in `seq 1 $nz`
z=1
while [ $z -le $nz ]
do
  # max of this slice
  zm=$((z+10))
  if [ $zm -gt $nz ]; then
    zm=$nz
  fi

  #z=$((tz-1))
  #z=$tz
  echo "  <Piece Extent=\"1 $nx 1 $ny $z $zm\" Source=\"slab_$zm.vti\"/>" >> pslab.pvti

  if [ $zm -lt $nz ]; then
    z=$((zm))
  else
    z=$((nz+1))
  fi
done

cat >> pslab.pvti <<EOF
 </PImageData>
</VTKFile>
EOF

#  <PImageData WholeExtent="1 $nx 1 $ny 1 $nz" Origin="0 0 0" Spacing="$dx $dy $dz" GhostLevel="0">
# echo "  <Piece Extent=\"1 $nx 1 $ny $z $tz\"  Source=\"slab_$z.vti\" />" >> pslab.pvti
