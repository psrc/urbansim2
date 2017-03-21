# PSRC Urbansim2 Makefile
#
# Usage:
# make       ---> just run using container's copy of code and date
# make latest --> fetch latest code from git and data from data directory
#
# What data directory, you ask? In your VirtualBox VM settings, add a new
# "Shared Folder" called "data" and point it to your data folder.
# Then launch Docker using:
#     docker run -v /data:/data urbansim2
#

UDST_REPOS=urbansim urbansim_defaults orca pandana

THIS_FILE := $(lastword $(MAKEFILE_LIST))

# Default make rule: simply run urbansim2 using container's copy of code & base data
default : simresult.h5

RUN_ID := $(shell bash get_run_id.sh)

simresult.h5 :
	@echo '--------'
	RUN_LOCATION=$(bash get_run_id.sh)
	python simulate_demo.py
	mv $@ "/data/$(RUN_ID).h5"


# latest: Pull all the latest code from github
# (clunky, invokes make again to build simresult.h5)

.PHONY : latest latestcode latestdata
latest : latestcode latestdata
	@$(MAKE) -f $(THIS_FILE) simresult.h5

latestdata :
	@echo '--------'
	cp /data/psrc_base_year_2014.h5 /udst/psrc_urbansim/data

latestcode : udst hana_edits

.PHONY : udst
udst :
	for each in $(UDST_REPOS); do \
		echo $$each ---------; \
		cd /udst/$$each; \
		git pull; \
		done


.PHONY : hana_edits
hana_edits :
	#cd /udst/urbansim_defaults
	#git remote add psrcedits https://github.com/hanase/urbansim_defaults.git
	#git pull psrcedits dev
	#
	#cd /udst/urbansim
	#git remote add psrcedits https://github.com/hanase/urbansim.git
	#git pull psrcedits dev


.PHONY: variables
variables:
	@echo UDST_REPOS: $(CODE)

