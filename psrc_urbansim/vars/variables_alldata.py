import pandas as pd
import numpy as np
import orca
from urbansim.utils import misc
import urbansim_defaults.utils

#####################
# ALLDATA VARIABLES (in alphabetic order)
#####################
#'full_time_worker','part_time_worker', 'non_working_adult_age_65_plus',
#'non_working_adult_age_16_64',
#'university_student','hs_student_age_15_up',
#'child_age_5_15','child_age_0_4'




@orca.column('alldata', 'age_0_to_5', cache=True, cache_scope='iteration')
def age_0_to_5(alldata, persons): 
    return pd.Series((persons.age <= 5).sum(), index = alldata.index)

@orca.column('alldata', 'age_6_to_10', cache=True, cache_scope='iteration')
def age_6_to_10(alldata, persons): 
    return pd.Series(((persons.age >= 6) * (persons.age <= 10)).sum(), index = alldata.index)

@orca.column('alldata', 'age_11_to_15', cache=True, cache_scope='iteration')
def age_11_to_15(alldata, persons): 
    return pd.Series(((persons.age >= 11) * (persons.age <= 15)).sum(), index = alldata.index)

@orca.column('alldata', 'age_16_to_20', cache=True, cache_scope='iteration')
def age_16_to_20(alldata, persons): 
    return pd.Series(((persons.age >= 16) * (persons.age <= 20)).sum(), index = alldata.index)

@orca.column('alldata', 'age_21_to_25', cache=True, cache_scope='iteration')
def age_21_to_25(alldata, persons): 
    return pd.Series(((persons.age >= 21) * (persons.age <= 25)).sum(), index = alldata.index)

@orca.column('alldata', 'age_26_to_30', cache=True, cache_scope='iteration')
def age_26_to_30(alldata, persons): 
    return pd.Series(((persons.age >= 26) * (persons.age <= 30)).sum(), index = alldata.index)

@orca.column('alldata', 'age_31_to_35', cache=True, cache_scope='iteration')
def age_31_to_35(alldata, persons): 
    return pd.Series(((persons.age >= 31) * (persons.age <= 35)).sum(), index = alldata.index)

@orca.column('alldata', 'age_36_to_40', cache=True, cache_scope='iteration')
def age_36_to_40(alldata, persons): 
    return pd.Series(((persons.age >= 36) * (persons.age <= 40)).sum(), index = alldata.index)

@orca.column('alldata', 'age_41_to_45', cache=True, cache_scope='iteration')
def age_41_to_45(alldata, persons): 
    return pd.Series(((persons.age >= 41) * (persons.age <= 45)).sum(), index = alldata.index)

@orca.column('alldata', 'age_46_to_50', cache=True, cache_scope='iteration')
def age_46_to_50(alldata, persons): 
    return pd.Series(((persons.age >= 46) * (persons.age <= 50)).sum(), index = alldata.index)

@orca.column('alldata', 'age_51_to_55', cache=True, cache_scope='iteration')
def age_51_to_55(alldata, persons): 
    return pd.Series(((persons.age >= 51) * (persons.age <= 55)).sum(), index = alldata.index)

@orca.column('alldata', 'age_56_to_60', cache=True, cache_scope='iteration')
def age_56_to_60(alldata, persons): 
    return pd.Series(((persons.age >= 56) * (persons.age <= 60)).sum(), index = alldata.index)

@orca.column('alldata', 'age_61_to_65', cache=True, cache_scope='iteration')
def age_61_to_65(alldata, persons): 
    return pd.Series(((persons.age >= 61) * (persons.age <= 65)).sum(), index = alldata.index)

@orca.column('alldata', 'age_66_to_70', cache=True, cache_scope='iteration')
def age_66_to_70(alldata, persons): 
    return pd.Series(((persons.age >= 66) * (persons.age <= 70)).sum(), index = alldata.index)

@orca.column('alldata', 'age_71_to_75', cache=True, cache_scope='iteration')
def age_71_to_75(alldata, persons): 
    return pd.Series(((persons.age >= 71) * (persons.age <= 75)).sum(), index = alldata.index)

@orca.column('alldata', 'age_76_to_80', cache=True, cache_scope='iteration')
def age_76_to_80(alldata, persons): 
    return pd.Series(((persons.age >= 76) * (persons.age <= 80)).sum(), index = alldata.index)

@orca.column('alldata', 'age_81_to_85', cache=True, cache_scope='iteration')
def age_81_to_85(alldata, persons): 
    return pd.Series(((persons.age >= 81) * (persons.age <= 85)).sum(), index = alldata.index)

@orca.column('alldata', 'age_86_to_90', cache=True, cache_scope='iteration')
def age_86_to_90(alldata, persons): 
    return pd.Series(((persons.age >= 86) * (persons.age <= 90)).sum(), index = alldata.index)

@orca.column('alldata', 'age_91_to_95', cache=True, cache_scope='iteration')
def age_91_to_95(alldata, persons): 
    return pd.Series(((persons.age >= 91) * (persons.age <= 95)).sum(), index = alldata.index)

@orca.column('alldata', 'age_96_and_up', cache=True, cache_scope='iteration')
def age_96_and_up(alldata, persons): 
    return pd.Series((persons.age >= 96).sum(), index = alldata.index)

@orca.column('alldata', 'Five_18', cache=True, cache_scope='iteration')
def Five_18(alldata, persons): 
    return pd.Series(((persons.age >= 5) * (persons.age <= 18)).sum(), index = alldata.index)

@orca.column('alldata', 'Group1_Under36870K', cache=True, cache_scope='iteration')
def Group1_Under36870K(alldata, households): 
    return pd.Series((households.income < 36870).sum(), index = alldata.index)

@orca.column('alldata', 'Group1_Under50K', cache=True, cache_scope='iteration')
def Group1_Under50K(alldata, households): 
    return pd.Series((households.income < 50000).sum(), index = alldata.index)

@orca.column('alldata', 'Group2_UpTo73700', cache=True, cache_scope='iteration')
def Group2_UpTo73700(alldata, households): 
    return pd.Series(((households.income > 36870) * (households.income < 73700)).sum(), index = alldata.index)

@orca.column('alldata', 'Group2_50_75K', cache=True, cache_scope='iteration')
def Group2_50_75K(alldata, households): 
    return pd.Series(((households.income > 50000) * (households.income < 75000)).sum(), index = alldata.index)

@orca.column('alldata', 'Group3_UpTo110600', cache=True, cache_scope='iteration')
def Group3_UpTo110600(alldata, households): 
    return pd.Series(((households.income > 73700) * (households.income < 110600)).sum(), index = alldata.index)

@orca.column('alldata', 'Group3_75_100K', cache=True, cache_scope='iteration')
def Group3_75_100K(alldata, households): 
    return pd.Series(((households.income > 75000) * (households.income < 100000)).sum(), index = alldata.index)

@orca.column('alldata', 'Group4_Over100K', cache=True, cache_scope='iteration')
def Group4_Over100K(alldata, households): 
    return pd.Series((households.income > 100000).sum(), index = alldata.index)

@orca.column('alldata', 'Group4_Over110600', cache=True, cache_scope='iteration')
def Group4_Over110600(alldata, households): 
    return pd.Series((households.income > 110600).sum(), index = alldata.index)

@orca.column('alldata', 'Nineteen_24', cache=True, cache_scope='iteration')
def Nineteen_24(alldata, persons): 
    return pd.Series(((persons.age >= 19) * (persons.age <= 24)).sum(), index = alldata.index)

@orca.column('alldata', 'number_of_households', cache=True, cache_scope='iteration')
def number_of_households(alldata, households):
    print 'in variables_alldata.py, in number_of_households function'
    return pd.Series(households.persons.size, index = alldata.index)

@orca.column('alldata', 'number_of_jobs', cache=True, cache_scope='iteration')
def number_of_jobs(alldata, jobs):
    print 'in variables_alldata.py, in number_of_jobs function'
    return pd.Series(jobs.number_of_jobs.sum(), index = alldata.index)

@orca.column('alldata', 'Over_60', cache=True, cache_scope='iteration')
def Over_60(alldata, persons): 
    return pd.Series((persons.age > 60).sum(), index = alldata.index)

@orca.column('alldata', 'population', cache=True, cache_scope='iteration')
def number_of_households(alldata, households):
    print 'in variables_alldata.py, in number_of_households function'
    return  pd.Series(households.persons.sum(), index = alldata.index)

@orca.column('alldata', 'residential_units', cache=True, cache_scope='iteration')
def residetial_units(alldata, buildings):
    print 'in variables_alldata.py, residential_units function'
    return pd.Series(buildings.residential_units.sum(), index = alldata.index)

@orca.column('alldata', 'Twentyfive_60', cache=True, cache_scope='iteration')
def Twentyfive_60(alldata, persons): 
    return pd.Series(((persons.age >= 25) * (persons.age <= 60)).sum(), index = alldata.index)

@orca.column('alldata', 'Under5', cache=True, cache_scope='iteration')
def Under5(alldata, persons): 
    return pd.Series((persons.age < 5).sum(), index = alldata.index)
