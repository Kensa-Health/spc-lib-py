# Generate XmR reference outputs with NHSRplotthedots 0.2.2.
# Install the exact reference source, not the moving default branch:
#   install.packages("remotes")
#   remotes::install_github("nhs-r-community/NHSRplotthedots@050b07d7eacc011767532732f69b8fd7d6a28fd3")
# Download the pinned input and run from the repository root:
#   curl -fL https://raw.githubusercontent.com/nhs-r-community/NHSRdatasets/defefde0927c189c81e2322a85be189c78f30686/data/ae_attendances.rda -o /tmp/ae_attendances.rda
#   Rscript scripts/generate_r_golden_output.R /tmp/ae_attendances.rda
# Review diffs and run tests/test_parity_xmr.py before accepting regenerated output.

library(NHSRplotthedots)
stopifnot(as.character(packageVersion("NHSRplotthedots")) == "0.2.2")
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Supply the pinned ae_attendances.rda path; see the script header")
}
input <- new.env()
load(args[[1]], envir = input)
dataset <- input$ae_attendances
subset <- dataset[dataset$org_code == "RJZ" & dataset$type == "1", ]
subset <- subset[order(subset$period), ]
stopifnot(nrow(subset) == 36L, !anyDuplicated(subset$period))

# Explicit defaults align with the Python single-pass, whole-segment calculation.
for (value_field in c("attendances", "breaches")) {
  direction <- if (value_field == "attendances") "neutral" else "decrease"
  result <- ptd_spc(
    subset,
    value_field = !!value_field,
    date_field = period,
    improvement_direction = direction,
    screen_outliers = TRUE
  )
  output <- as.data.frame(result[, c(
    "x", "y", "mean_col", "upl", "lpl", "outside_limits",
    "relative_to_mean", "close_to_limits", "special_cause_type", "point_type"
  )])
  output$x <- format(output$x, "%Y-%m-%d", tz = "UTC")
  filename <- if (value_field == "attendances") {
    "ae_attendances_rjz_type1_xmr_expected.csv"
  } else {
    "ae_attendances_rjz_type1_breaches_xmr_expected.csv"
  }
  write.csv(output, file.path("tests/golden/nhsrdatasets", filename), row.names = FALSE)
}
cat("Generated two fixtures with NHSRplotthedots", as.character(packageVersion("NHSRplotthedots")), "\n")
cat(R.version.string, "\n")
for (package in c("dplyr", "rlang", "stringr")) {
  cat(package, as.character(packageVersion(package)), "\n")
}
