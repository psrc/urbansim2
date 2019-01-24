library(data.table)
setwd("/Users/hana/udst/psrc_urbansim/data/zoning")

# read constraints
constr <- fread("development_constraints.csv")

# read the heights table exported from MS-SQL as csv
heights <- fread("heights.csv")

# # replace NULLs in max_coverage with -1
heights[max_coverage == "NULL", max_coverage := NA]
heights[, max_coverage := as.double(max_coverage)]
heights[is.na(max_coverage), max_coverage := -1]

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
plantypes <- copy(wconstr)
# merge residential into one
plantypes[, residential := as.integer((single_family_residential + multi_family_residential) > 0)]
plantypes[,`:=`(single_family_residential = NULL, multi_family_residential = NULL)]
plantypes <- merge(plantypes, heights, by = "plan_type_id")

# rename columns so that opus can translate it into the right type
column.names <- list(plan_type_id = "plan_type_id:i4",
                      max_du_ac = "max_du:f4",
                      max_far = "max_far:f4",
                      max_coverage = "max_coverage:f4",
                      max_height_ft = "max_height:f4",
                      height_imputed = "height_imputed:i4")
for(col in names(column.names))
  setnames(plantypes, col, column.names[[col]])
for(col in c("office", "commercial", "industrial", "mixed_use", "residential"))
  setnames(plantypes, col, paste0(col, ":i4"))

fwrite(plantypes, file = "zoning_heights.csv")
