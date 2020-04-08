#!/bin/bash
	VOPONOI_PLUGIN_RUN_STR="TBG_voronoi="--e_randomizedMerger.period 100 --e_randomizedMerger.minParticlesToMerge 100""
	pic-create $PIC_EXAMPLES/KelvinHelmholtz $HOME/picInputs/KelvinHelmholtz_not_reducted
	cd KelvinHelmholtz_not_reducted
	pic-build 
	tbg -f -s bash -t etc/picongpu/hemera-hzdr/k80.tpl -c etc/picongpu/1.cfg $SCRATCH/KelvinHelmholtz_not_reducted
	cd ..
	pic-create $PIC_EXAMPLES/KelvinHelmholtz $HOME/picInputs/KelvinHelmholtzreducted
	echo $VOPONOI_PLUGIN_RUN_STR | cat - '"/picInputs/KelvinHelmholtzreducted"1.cfg' > temp && mv temp '"/picInputs/KelvinHelmholtzreducted"1.cfg'
	PLUGIN_NAME="TBG_voronoi"
	filename=$HOME"/picInputs/KelvinHelmholtz/etc/picongpu/1.cfg"
	NEW_SGF_FILE ="/picInputs/KelvinHelmholtz/etc/picongpu/KelvinHelmholtzreducted1.cfg"
    for line in $(cat < $filename); do
						if [[ $line = *'TBG_plugins="'* ]]; then
								echo $PLUGIN_NAME
								echo $PLUGIN_NAME >> $NEW_SGF_FILE
						else
								echo  $line >> $NEW_SGF_FILE
						fi
	done
	tbg -f -s bash -t etc/picongpu/hemera-hzdr/k80.tpl -c etc/picongpu/1.cfg $SCRATCH/KelvinHelmholtzreducted
	
	