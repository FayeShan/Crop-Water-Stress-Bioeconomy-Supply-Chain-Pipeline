##############################################################################
# Fig: Four vertically-stacked area charts (1995–2024)
#   Panel 1: stacked by crop
#   Panel 2: stacked by production country
#   Panel 3: stacked by consumption country
#   Panel 4: stacked by target sector
#
# Stack order (top -> bottom) now matches the sankey (left -> right):
#   - crop / sector : 2024 total value, descending
#   - production / consumption country : unified (vp + vc) descending,
#     identical to the sankey's country_order
#   - each panel stacks independently via a panel-specific key, so shared
#     names (Rice / Fruits & nuts / Vegetables) no longer collide.
#
# This version creates two scopes:
#   1. total = all flows, impact_type == "total"
#   2. trade = international trade only, impact_type == "total",
#              production_country != consumption_country
#
# It also saves:
#   - stacked_area_data_long_total_trade.csv
#   - stacked_area_percent_long_total_trade.csv
##############################################################################

library(dplyr)
library(tidyr)
library(ggplot2)

source("config.R")

# ─────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────
df_raw <- read.csv(
  AGG_CSV,
  stringsAsFactors = FALSE
)

df_base <- df_raw %>%
  filter(
    impact_type == "total",
    year >= 1995,
    year <= 2024
  )

# total: all domestic + trade flows
df_total <- df_base %>%
  mutate(scope = "total")

# trade: only production country != consumption country
df_trade <- df_base %>%
  filter(production_country != consumption_country) %>%
  mutate(scope = "trade")

df <- bind_rows(df_total, df_trade)

# ─────────────────────────────────────────────────────────────────────────
# OUTPUT FOLDER
# ─────────────────────────────────────────────────────────────────────────
outdir <- file.path(OUTPUT_DIR, "stacked")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ─────────────────────────────────────────────────────────────────────────
# PALETTES
# ─────────────────────────────────────────────────────────────────────────
PAL_REGION <- c(
  "India"                              = "#1F4E79",
  "China"                              = "#C1272D",
  "Pakistan"                           = "#2E86AB",
  "Other Asia-Pacific"                 = "#8E6E3C",
  "Other Middle East & North Africa"   = "#E8A33D",
  "Egypt"                              = "#71a836",
  "Iran"                               = "#4B9F7E",
  "Europe"                             = "#6A4C93",
  "Other America"                      = "#7FB3D5",
  "Sub-Saharan Africa"                 = "#B84A8A",
  "United States"                      = "#2C3E75"
)

PAL_CROP <- c(
  "Wheat"         = "#E76F00",
  "Rice"          = "#D9C75F",
  "Maize"         = "#FFD23F",
  "Other Cereals" = "#8A9A32",
  "Fibre Crops"   = "#E63946",
  "Oil Seeds"     = "#4F8A10",
  "Sugar Crops"   = "#C9A227",
  "Fruits & nuts" = "#D1495B",
  "Vegetables"    = "#2A9D3F",
  "Livestock"     = "#8B5E34",
  "Other Crops"   = "#B5651D"
)

PAL_SECTOR <- c(
  "Cereals"                  = "#F9C80E",
  "Rice"                     = "#F86624",
  "Fibre & Textile"          = "#ae138c",
  "Oil products"             = "#277DA1",
  "Sugar & chocolate"        = "#6D4C41",
  "Fruits & nuts"            = "#f388b1",
  "Vegetables"               = "#70E000",
  "Other agri-food products" = "#F8961E",
  "Meat & fish"              = "#B5172F",
  "Dairy"                    = "#A7C7E7",
  "Beverages"                = "#3A0CA3",
  "Biochemicals"             = "#00B4D8"
)

ALL_PAL <- c(PAL_CROP, PAL_REGION, PAL_SECTOR)

# ─────────────────────────────────────────────────────────────────────────
# BUILD LONG TABLE FOR STACKED AREA
# one row per scope, panel, year, category
# ─────────────────────────────────────────────────────────────────────────
agg <- function(d, col, panel_label) {
  d %>%
    group_by(scope, year, cat = .data[[col]]) %>%
    summarise(value = sum(value, na.rm = TRUE), .groups = "drop") %>%
    mutate(panel = panel_label)
}

ts_crop <- agg(df, "crop",                "Crop")
ts_prod <- agg(df, "production_country",  "Production country")
ts_cons <- agg(df, "consumption_country", "Consumption country")
ts_sect <- agg(df, "target_sector",       "Target sector")

ts_all <- bind_rows(ts_crop, ts_prod, ts_cons, ts_sect)

ts_all$scope <- factor(
  ts_all$scope,
  levels = c("total", "trade")
)

ts_all$panel <- factor(
  ts_all$panel,
  levels = c(
    "Crop",
    "Production country",
    "Consumption country",
    "Target sector"
  )
)

# Save raw long table used for plotting
write.csv(
  ts_all,
  file.path(outdir, "stacked_area_data_long_total_trade.csv"),
  row.names = FALSE
)

cat(
  "Saved:",
  file.path(outdir, "stacked_area_data_long_total_trade.csv"),
  "\n"
)

# ─────────────────────────────────────────────────────────────────────────
# CREATE PERCENTAGE LONG TABLE
# percentage is within each scope + panel + year
# ─────────────────────────────────────────────────────────────────────────
label_percent_table <- ts_all %>%
  group_by(scope, panel, year) %>%
  mutate(
    total_value = sum(value, na.rm = TRUE),
    percent = ifelse(total_value == 0, NA, value / total_value * 100),
    percent_label = paste0(round(percent, 2), "%")
  ) %>%
  ungroup() %>%
  arrange(scope, panel, year, desc(percent))

write.csv(
  label_percent_table,
  file.path(outdir, "stacked_area_percent_long_total_trade.csv"),
  row.names = FALSE
)

cat(
  "Saved:",
  file.path(outdir, "stacked_area_percent_long_total_trade.csv"),
  "\n"
)

# Optional: also save separate total/trade percent long tables
write.csv(
  label_percent_table %>% filter(scope == "total"),
  file.path(outdir, "stacked_area_percent_long_total.csv"),
  row.names = FALSE
)

write.csv(
  label_percent_table %>% filter(scope == "trade"),
  file.path(outdir, "stacked_area_percent_long_trade.csv"),
  row.names = FALSE
)

cat(
  "Saved:",
  file.path(outdir, "stacked_area_percent_long_total.csv"),
  "\n"
)

cat(
  "Saved:",
  file.path(outdir, "stacked_area_percent_long_trade.csv"),
  "\n"
)

# ─────────────────────────────────────────────────────────────────────────
# STACK ORDER  (matches the sankey: largest first, per panel, independently)
#
# IMPORTANT: the order is computed PER SCOPE.
#   - total figure  -> ordered by total-scope 2024 values  (= sankey Fig 4a)
#   - trade figure  -> ordered by trade-scope 2024 values  (= sankey Fig 4b)
# Using one global (total-only) order made the trade figure ignore its own
# trade-based ranking, so it never matched sankey 4b.
# ─────────────────────────────────────────────────────────────────────────
yr <- max(ts_all$year)

# Raw "panel@@cat" key on every row; levels are assigned per-scope at plot time.
ts_all$stack_key <- paste(ts_all$panel, ts_all$cat, sep = "@@")

# Build the panel-specific level ordering for a given scope.
# A panel-specific stacking key lets each panel stack INDEPENDENTLY: a single
# global factor on `cat` cannot work, because country names repeat across the
# two country panels, and Rice / Fruits & nuts / Vegetables repeat across crop
# & sector — unique() would collapse them and scramble the order.
make_panel_levels <- function(scope_ref) {

  # crop & sector: this scope's 2024 value, descending (sankey arrange(desc(v)))
  ord_crop <- ts_crop %>% filter(scope == scope_ref, year == yr) %>%
    group_by(cat) %>% summarise(v = sum(value, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(v)) %>% pull(cat)

  ord_sect <- ts_sect %>% filter(scope == scope_ref, year == yr) %>%
    group_by(cat) %>% summarise(v = sum(value, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(v)) %>% pull(cat)

  # country: unified (vp + vc) descending — replicates the sankey country_order
  vp <- ts_prod %>% filter(scope == scope_ref, year == yr) %>%
    group_by(cat) %>% summarise(vp = sum(value, na.rm = TRUE), .groups = "drop")
  vc <- ts_cons %>% filter(scope == scope_ref, year == yr) %>%
    group_by(cat) %>% summarise(vc = sum(value, na.rm = TRUE), .groups = "drop")

  country_ord <- full_join(vp, vc, by = "cat") %>%
    mutate(vp = coalesce(vp, 0), vc = coalesce(vc, 0), v_total = vp + vc) %>%
    arrange(desc(v_total)) %>% pull(cat)

  ord_prod <- country_ord[country_ord %in% unique(ts_prod$cat)]
  ord_cons <- country_ord[country_ord %in% unique(ts_cons$cat)]

  # Append any categories not present in this scope's 2024 so nothing is dropped
  ord_crop <- c(as.character(ord_crop), setdiff(unique(ts_crop$cat), ord_crop))
  ord_sect <- c(as.character(ord_sect), setdiff(unique(ts_sect$cat), ord_sect))
  ord_prod <- c(as.character(ord_prod), setdiff(unique(ts_prod$cat), ord_prod))
  ord_cons <- c(as.character(ord_cons), setdiff(unique(ts_cons$cat), ord_cons))

  c(
    paste("Crop",                ord_crop, sep = "@@"),
    paste("Production country",  ord_prod, sep = "@@"),
    paste("Consumption country", ord_cons, sep = "@@"),
    paste("Target sector",       ord_sect, sep = "@@")
  )
}

# Fallback grey for categories without a palette color
missing <- setdiff(unique(as.character(ts_all$cat)), names(ALL_PAL))
if (length(missing) > 0) {
  ALL_PAL <- c(
    ALL_PAL,
    setNames(rep("#CCCCCC", length(missing)), missing)
  )
}

ALL_PAL_FILL <- scales::alpha(ALL_PAL, 0.6)
ALL_PAL_LINE <- scales::alpha(ALL_PAL, 1.0)

# ─────────────────────────────────────────────────────────────────────────
# PLOT FUNCTION
# Save one stacked area chart for total and one for trade
# ─────────────────────────────────────────────────────────────────────────
make_stacked_area <- function(plot_data, scope_name) {

  # Assign this scope's own stacking order (total -> sankey 4a, trade -> 4b)
  plot_data$stack_key <- factor(
    plot_data$stack_key,
    levels = make_panel_levels(scope_name)
  )

  p <- ggplot(plot_data, aes(x = year, y = value,
                             group = stack_key, fill = cat, colour = cat)) +
    # If the largest ends up on the BOTTOM and you want it on top (or vice
    # versa), add: position = position_stack(reverse = TRUE)
    geom_area(linewidth = 0.18) +
    scale_fill_manual(values = ALL_PAL_FILL, guide = "none") +
    scale_colour_manual(values = ALL_PAL_LINE, guide = "none") +
    scale_x_continuous(
      breaks = c(1995, 2005, 2015, 2024),
      limits = c(1995, 2024),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      expand = c(0, 0),
      labels = scales::label_number(scale_cut = scales::cut_short_scale())
    ) +
    facet_wrap(
      ~ panel,
      ncol = 1,
      scales = "free_y",
      strip.position = "top"
    ) +
    theme_minimal(base_size = 10) +
    theme(
      strip.text = element_blank(),
      strip.background   = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(colour = "#EEEEEE", linewidth = 0.3),
      axis.title         = element_blank(),
      axis.ticks.y       = element_line(colour = "#000000", linewidth = 0.3),
      axis.ticks.length  = unit(2, "pt"),
      axis.text.y        = element_text(colour = "#000000", size = 8),
      axis.line.y        = element_line(colour = "#000000", linewidth = 0.3),
      axis.text.x        = element_text(colour = "#000000", size = 9),
      axis.ticks.x       = element_line(colour = "#000000", linewidth = 0.3),
      axis.line.x        = element_line(colour = "#000000", linewidth = 0.3),
      panel.spacing.y    = unit(8, "pt"),
      plot.margin        = margin(6, 10, 6, 6)
    )

  ggsave(
    file.path(outdir, paste0("timeseries_stacked_areas_", scope_name, ".pdf")),
    p,
    width = 3,
    height = 10,
    units = "in"
  )

  ggsave(
    file.path(outdir, paste0("timeseries_stacked_areas_", scope_name, ".png")),
    p,
    width = 3,
    height = 10,
    units = "in",
    dpi = 300
  )

  cat(
    "Saved:",
    file.path(outdir, paste0("timeseries_stacked_areas_", scope_name, ".pdf")),
    "\n"
  )

  cat(
    "Saved:",
    file.path(outdir, paste0("timeseries_stacked_areas_", scope_name, ".png")),
    "\n"
  )
}

# ─────────────────────────────────────────────────────────────────────────
# SAVE FIGURES
# ─────────────────────────────────────────────────────────────────────────
make_stacked_area(
  ts_all %>% filter(scope == "total"),
  "total"
)

make_stacked_area(
  ts_all %>% filter(scope == "trade"),
  "trade"
)

cat("\nDone! Total and trade stacked area figures and long tables saved.\n")