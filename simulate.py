import models
import urbansim.sim.simulation as sim
sim.run([
    "repm_simulate",              # REPM
    #"nrh_simulate",              # non-residential rent hedonic

    #"households_relocation",     # households relocation model
    #"hlcm_simulate",            # households location choice
    #"households_transition",     # households transition

    #"jobs_relocation",           # jobs relocation model
    #"elcm_simulate",             # employment location choice
    #"jobs_transition",           # jobs transition

    #"feasibility",               # compute development feasibility
    #"residential_developer",     # build residential buildings
    #"non_residential_developer", # build non-residential buildings
], years=[2001, 2002], data_out="simresults.h5")