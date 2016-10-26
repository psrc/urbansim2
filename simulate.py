import psrc_urbansim.models
import orca
import logging

#logging.basicConfig(level=logging.DEBUG)

orca.run([
    #"repmres_simulate",           # residential REPM
    #"repmnr_simulate",            # non-residential REPM

    #"households_relocation",     # households relocation model
    #"hlcm_simulate",              # households location choice
    #"households_transition",     # households transition

    #"jobs_relocation",           # jobs relocation model
    #"elcm_simulate",             # employment location choice
    "jobs_transition",           # jobs transition

    #"feasibility",               # compute development feasibility
    #"residential_developer",     # build residential buildings
    #"non_residential_developer", # build non-residential buildings
], iter_vars=[2015, 2016], data_out="simresult.h5")

logging.info('Simulation finished')