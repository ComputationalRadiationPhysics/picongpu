all: species_final.png species_initial.png species_prebook.png species_final_detail.png species_initial_detail.png species_example_init.png species_example_final.png

species_final_detail.png: species.plantuml
	plantuml -p -DDETAIL=1 species.plantuml <$< >$@

species_final.png: species.plantuml
	plantuml -p species.plantuml <$< >$@

species_initial.png: species.plantuml
	plantuml -p -DSETUP=1 species.plantuml <$< >$@

species_initial_detail.png: species.plantuml
	plantuml -p -DDETAIL=1 -DSETUP=1 species.plantuml <$< >$@

species_prebook.png: species.plantuml
	plantuml -p -DPREBOOK=1 species.plantuml <$< >$@

species_example_init.png: species_example.plantuml
	plantuml -p species_example.plantuml <$< >$@

species_example_final.png: species_example.plantuml
	plantuml -p -DSHOWATTRS=1 species_example.plantuml <$< >$@
