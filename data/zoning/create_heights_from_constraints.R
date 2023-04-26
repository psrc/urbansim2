# Script for creating the zoning_heights table.
# The info is extracted from development_constraints table
# which was created using the scripts
# https://github.com/psrc/urbansim-baseyear-prep/blob/master/future_land_use/development_constraints_imputation.R
# and then
# https://github.com/psrc/urbansim-baseyear-prep/blob/master/future_land_use/unroll_constraints.py
# The input development_constraints table should contain column maxht (height) and lc (coverage).
#
# Hana Sevcikova (PSRC)
# 2023/04/10

library(data.table)

# read the constraints table (which includes coverage and max height)
#constr.file <- "devconstr_2021-06-17.csv"
constr.file <- "~/n$/base_year_2018_inputs/urbansim2_cache/additional_tables/devconstr_v2_2023-01-10.csv"

constr <- fread(constr.file)
constr[, maxht := pmax(maxht, 12)] # minimum height

# convert constraints into wide format
wconstr <- dcast(constr, plan_type_id ~ generic_land_use_type_id , value.var = "maximum", 
                 fun.aggregate = max, fill = NA)

# rename columns
glut <- c("single_family_residential", # 1
          "multi_family_residential", # 2
          "office", # 3
          "commercial", # 4
          "industrial", # 5
          "mixed_use" # 6
)
colnames(wconstr)[-1] <- glut
# replace values with 1 and 0 (allowed and not allowed)
for(col in colnames(wconstr)[-1]) {
    wconstr[[col]][is.na(wconstr[[col]])] <- 0
    wconstr[[col]][wconstr[[col]] > 0] <- 1
}

htlc <- constr[, .(max_height = min(maxht), max_coverage = min(lc)), by = plan_type_id]

dua <- constr[constraint_type == "units_per_acre", .(max_du = min(maximum[maximum > 0])), by = plan_type_id]
dua[is.infinite(max_du), max_du := 0]
far <- constr[constraint_type == "far", .(max_far = min(maximum[maximum > 0])), by = plan_type_id]
far[is.infinite(max_far), max_far := 0]


allconstr <- merge(wconstr, merge(merge(dua, far, by = "plan_type_id"), htlc, by = "plan_type_id"), by = "plan_type_id")

# rename columns so that opus can translate it into the right type
fcols <- c("max_du", "max_far", "max_coverage", "max_height") # float type
icols <- setdiff(colnames(allconstr), fcols)

for(col in fcols)
    setnames(allconstr, col, paste0(col, ":f4"))
# everything else is integer
for(col in icols)
    setnames(allconstr, col, paste0(col, ":i4"))


fwrite(allconstr, file = "zoning_heights.csv")

# export to Opus cache with 
# python -m opus_core.tools.convert_table csv flt -d . -o  {dir}/opusgit/urbansim_data/data/psrc_parcel/base_year_2018_inputs/urbansim2_cache/2018 -t zoning_heights

