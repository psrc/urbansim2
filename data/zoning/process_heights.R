# Script for processing the zoning_heights table 
#
# Hana Sevcikova (PSRC)
# 2019/01/16

library(data.table)

# read the heights table exported from MS-SQL as csv
heights <- fread("heights.csv")

# replace NULLs in max_coverage with -1
heights[max_coverage == "NULL", max_coverage := NA]
heights[, max_coverage := as.double(max_coverage)]
heights[is.na(max_coverage), max_coverage := -1]

# rename columns so that opus can translate it into teh right type
column.names <- list(plan_type_id = "plan_type_id:i4", 
                    max_du_ac = "max_du:f4",
                    max_far = "max_far:f4",
                    max_coverage = "max_coverage:f4",
                    max_height_ft = "max_height:f4",
                    height_imputed = "height_imputed:b1")
for(col in names(column.names))
  setnames(heights, col, column.names[[col]])

fwrite(heights, file = "zoning_heights.csv")

# export to Opus cache with 
# python -m opus_core.tools.convert_table csv flt -d . -o  {dir}/opusgit/urbansim_data/data/psrc_parcel/base_year_2014_inputs/urbansim2_cache/2014 -t zoning_heights

##########
## Experimetnal code if comparison to the contraints table desired
#constraints <- fread("development_constraints.csv")
## convert constraints into wide format
#wconstr <- dcast(constraints, plan_type_id ~ constraint_type , value.var = "maximum", fun.aggregate = max)
## compute heights
#wconstr[, max_height := pmax((20 * far), (15 * sqrt(units_per_acre)),35)]
## replace inf and NA with 0
#wconstr[is.infinite(far) | is.na(far), far := 0]
#wconstr[is.infinite(units_per_acre) | is.na(units_per_acre), units_per_acre := 0]
## rename columns
#setnames(wconstr, "far", "max_far")
#setnames(wconstr, "units_per_acre", "max_dua")
## replace max_far 
#missing.heights <- unique(constraints[!plan_type_id %in% heights$plan_type_id, plan_type_id])
## merge with heights
#tmp <- merge(heights, wconstr, by = "plan_type_id")





