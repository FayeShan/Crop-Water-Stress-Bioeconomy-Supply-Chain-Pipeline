##############################################################################
# Export label percentage tables matching Fig 4a global and Fig 4b trade
##############################################################################

library(dplyr)
library(tidyr)

source("config.R")

df <- read.csv(
  AGG_CSV,
  stringsAsFactors = FALSE
)

# Use exactly the same base filter as the Sankey code
df_2024 <- df %>%
  filter(
    year == 2024,
    impact_type == "total"
  )

# Same as Fig 4a: global
df_global <- df_2024

# Same as Fig 4b: trade only
df_trade <- df_2024 %>%
  filter(production_country != consumption_country)

outdir <- file.path(OUTPUT_DIR, "sankeys")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ─────────────────────────────────────────────────────────────────────────
# Helper: calculate percentage table for one dataset
# ─────────────────────────────────────────────────────────────────────────
make_percent_table <- function(data, scope_name) {
  
  total_val <- sum(data$value, na.rm = TRUE)
  
  crop_tab <- data %>%
    group_by(cat = crop) %>%
    summarise(value = sum(value, na.rm = TRUE), .groups = "drop") %>%
    mutate(panel = "Crop")
  
  prod_tab <- data %>%
    group_by(cat = production_country) %>%
    summarise(value = sum(value, na.rm = TRUE), .groups = "drop") %>%
    mutate(panel = "Production country")
  
  cons_tab <- data %>%
    group_by(cat = consumption_country) %>%
    summarise(value = sum(value, na.rm = TRUE), .groups = "drop") %>%
    mutate(panel = "Consumption country")
  
  sector_tab <- data %>%
    group_by(cat = target_sector) %>%
    summarise(value = sum(value, na.rm = TRUE), .groups = "drop") %>%
    mutate(panel = "Target sector")
  
  bind_rows(crop_tab, prod_tab, cons_tab, sector_tab) %>%
    mutate(
      scope = scope_name,
      total_value = total_val,
      percent = value / total_value * 100,
      percent_label = paste0(round(percent, 1), "%")
    ) %>%
    select(scope, panel, cat, value, total_value, percent, percent_label) %>%
    arrange(scope, panel, desc(percent))
}

# Build global and trade tables
tab_global <- make_percent_table(df_global, "global")
tab_trade  <- make_percent_table(df_trade,  "trade")

tab_all <- bind_rows(tab_global, tab_trade)

# ─────────────────────────────────────────────────────────────────────────
# Save long tables
# ─────────────────────────────────────────────────────────────────────────
write.csv(
  tab_global,
  file.path(outdir, "fig4a_global_label_percent_long.csv"),
  row.names = FALSE
)

write.csv(
  tab_trade,
  file.path(outdir, "fig4b_trade_label_percent_long.csv"),
  row.names = FALSE
)

write.csv(
  tab_all,
  file.path(outdir, "fig4_global_and_trade_label_percent_long.csv"),
  row.names = FALSE
)

# ─────────────────────────────────────────────────────────────────────────
# Save wide table: easier to inspect in Excel
# ─────────────────────────────────────────────────────────────────────────
tab_all_wide <- tab_all %>%
  select(scope, panel, cat, percent) %>%
  pivot_wider(
    names_from = cat,
    values_from = percent,
    values_fill = 0
  ) %>%
  arrange(scope, panel)

write.csv(
  tab_all_wide,
  file.path(outdir, "fig4_global_and_trade_label_percent_wide.csv"),
  row.names = FALSE
)

cat("Saved percentage tables to:\n")
cat(file.path(outdir, "fig4a_global_label_percent_long.csv"), "\n")
cat(file.path(outdir, "fig4b_trade_label_percent_long.csv"), "\n")
cat(file.path(outdir, "fig4_global_and_trade_label_percent_long.csv"), "\n")
cat(file.path(outdir, "fig4_global_and_trade_label_percent_wide.csv"), "\n")