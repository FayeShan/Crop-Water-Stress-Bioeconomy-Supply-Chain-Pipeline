##############################################################################
# Fig 5 composition plots (100% stacked bar) — restyled to match the Fig 4
# timeseries stacked look (semi-transparent fill + same-colour line, black
# axes, stacking order aligned to Fig 4).
#
# 6 figures, each output independently:
#   Fibre & Textile: by crop / production region / consumption region
#   Meat & fish:     by crop / production region / consumption region
#
# Stacking order (top -> bottom) matches Fig 4:
#   - crop : 2024 value, descending
#   - production / consumption region : unified (vp + vc) descending
#   Largest sits on TOP (position_fill(reverse = TRUE)).
##############################################################################

library(dplyr)
library(ggplot2)
library(scales)

source("config.R")

# ─────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────
df <- read.csv(AGG_CSV,
               stringsAsFactors = FALSE)

# Keep total only
df <- df %>%
  filter(impact_type == "total")

# Trade only
df <- df %>%
  filter(production_country != consumption_country)


# ─────────────────────────────────────────────────────────────────────────
# PALETTES
# ─────────────────────────────────────────────────────────────────────────
PAL_REGION <- c(
  "India"                            = "#1F4E79",
  "China"                            = "#C1272D",
  "Pakistan"                         = "#2E86AB",
  "Other Asia-Pacific"               = "#8E6E3C",
  "Other Middle East & North Africa" = "#E8A33D",
  "Egypt"                            = "#71A836",
  "Iran"                             = "#4B9F7E",
  "Europe"                           = "#6A4C93",
  "Other America"                    = "#7FB3D5",
  "Sub-Saharan Africa"               = "#B84A8A",
  "United States"                    = "#2C3E75"
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

# ─────────────────────────────────────────────────────────────────────────
# HELPER: complete palette if some categories are missing from palette
# ─────────────────────────────────────────────────────────────────────────
complete_palette <- function(levels_needed, base_pal) {
  missing_lvls <- setdiff(levels_needed, names(base_pal))
  if (length(missing_lvls) == 0) return(base_pal[levels_needed])

  extra_cols <- setNames(hcl.colors(length(missing_lvls), "Set 3"), missing_lvls)
  pal <- c(base_pal, extra_cols)
  pal[levels_needed]
}

# ─────────────────────────────────────────────────────────────────────────
# HELPER: stacking / legend order aligned to Fig 4
#   - crop, consumption-only, etc.: just rank by 2024 value desc
#   - production_country / consumption_country: unified (vp + vc) desc so the
#     same region sits in the same stack position across both views, exactly
#     like Fig 4's country_order.
# ─────────────────────────────────────────────────────────────────────────
fig4_order <- function(data, group_var) {
  yr <- max(data$year)

  if (group_var %in% c("production_country", "consumption_country")) {
    vp <- data %>% filter(year == yr) %>%
      group_by(cat = production_country) %>%
      summarise(vp = sum(value, na.rm = TRUE), .groups = "drop")
    vc <- data %>% filter(year == yr) %>%
      group_by(cat = consumption_country) %>%
      summarise(vc = sum(value, na.rm = TRUE), .groups = "drop")

    ord <- full_join(vp, vc, by = "cat") %>%
      mutate(vp = coalesce(vp, 0), vc = coalesce(vc, 0), v_total = vp + vc) %>%
      arrange(desc(v_total)) %>% pull(cat)
  } else {
    ord <- data %>% filter(year == yr) %>%
      group_by(cat = .data[[group_var]]) %>%
      summarise(v = sum(value, na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(v)) %>% pull(cat)
  }
  ord
}

# ─────────────────────────────────────────────────────────────────────────
# HELPER: build one 100% stacked bar chart, Fig 4 styling
# group_var: "crop", "production_country", "consumption_country"
# sector_name: e.g. "Fibre & Textile", "Meat & fish"
# ─────────────────────────────────────────────────────────────────────────
make_comp_plot <- function(data,
                           sector_name,
                           group_var,
                           palette,
                           fname,
                           outdir = file.path(OUTPUT_DIR, "fig5_composition"),
                           width = 14,
                           height = 10,
                           show_legend = TRUE) {

  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

  # Filter one target sector
  df_sub <- data %>%
    filter(target_sector == sector_name)

  # Summarise by year + grouping variable
  plot_df <- df_sub %>%
    group_by(year, .data[[group_var]]) %>%
    summarise(value = sum(value), .groups = "drop") %>%
    rename(group = .data[[group_var]])

  # Stacking / legend order aligned to Fig 4 (unified region order, 2024-based)
  group_order <- fig4_order(df_sub, group_var)
  # keep only groups that actually appear, append any stragglers
  group_order <- c(
    group_order[group_order %in% unique(plot_df$group)],
    setdiff(unique(plot_df$group), group_order)
  )

  plot_df$group <- factor(plot_df$group, levels = group_order)

  pal_use      <- complete_palette(group_order, palette)
  pal_fill     <- scales::alpha(pal_use, 0.6)   # Fig 4: semi-transparent fill
  pal_line     <- scales::alpha(pal_use, 1.0)   # Fig 4: opaque same-colour line

  # legend title
  fill_title <- dplyr::case_when(
    group_var == "crop" ~ "Crop",
    group_var == "production_country" ~ "Production region",
    group_var == "consumption_country" ~ "Consumption region",
    TRUE ~ group_var
  )

  p <- ggplot(plot_df, aes(x = factor(year), y = value,
                           fill = group, colour = group)) +
    geom_col(
      position = position_stack(reverse = TRUE),  # absolute stack, largest on TOP — like Fig 4
      width = 0.95,
      linewidth = 0.20
    ) +
    scale_y_continuous(
      labels = scales::label_number(scale_cut = scales::cut_short_scale()),
      expand = c(0, 0)
    ) +
    scale_fill_manual(values = pal_fill, drop = FALSE, name = fill_title) +
    scale_colour_manual(values = pal_line, drop = FALSE, guide = "none") +
    labs(
      x = NULL,
      y = NULL,
      title = NULL
    ) +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(colour = "#EEEEEE", linewidth = 0.3),
      axis.text.x        = element_text(size = 9,  colour = "#000000"),
      axis.text.y        = element_text(size = 11, colour = "#000000"),
      axis.title.y       = element_text(size = 12, colour = "#000000"),
      axis.ticks         = element_line(colour = "#000000", linewidth = 0.3),
      axis.ticks.length  = unit(2, "pt"),
      axis.line          = element_line(colour = "#000000", linewidth = 0.3),
      plot.title         = element_blank(),
      legend.title       = element_text(size = 11, colour = "#000000"),
      legend.text        = element_text(size = 10, colour = "#000000"),
      legend.position    = ifelse(show_legend, "right", "none"),
      plot.background    = element_rect(fill = "white", colour = NA),
      panel.background   = element_rect(fill = "white", colour = NA)
    )

  ggsave(file.path(outdir, paste0(fname, ".pdf")),
         p, width = width, height = height, units = "in")

  ggsave(file.path(outdir, paste0(fname, ".png")),
         p, width = width, height = height, units = "in", dpi = 300)

  cat("Saved:", fname, "\n")
  return(p)
}

# ─────────────────────────────────────────────────────────────────────────
# GENERATE 6 FIGURES
# ─────────────────────────────────────────────────────────────────────────

# 1) Fibre & Textile — by crop
p1 <- make_comp_plot(
  data = df,
  sector_name = "Fibre & Textile",
  group_var = "crop",
  palette = PAL_CROP,
  fname = "fig5_fibre_trade_by_crop"
)

# 2) Fibre & Textile — by production region
p2 <- make_comp_plot(
  data = df,
  sector_name = "Fibre & Textile",
  group_var = "production_country",
  palette = PAL_REGION,
  fname = "fig5_fibre_trade_by_production_region"
)

# 3) Fibre & Textile — by consumption region
p3 <- make_comp_plot(
  data = df,
  sector_name = "Fibre & Textile",
  group_var = "consumption_country",
  palette = PAL_REGION,
  fname = "fig5_fibre_trade_by_consumption_region"
)

# 4) Meat & fish — by crop
p4 <- make_comp_plot(
  data = df,
  sector_name = "Meat & fish",
  group_var = "crop",
  palette = PAL_CROP,
  fname = "fig5_meat_trade_by_crop"
)

# 5) Meat & fish — by production region
p5 <- make_comp_plot(
  data = df,
  sector_name = "Meat & fish",
  group_var = "production_country",
  palette = PAL_REGION,
  fname = "fig5_meat_trade_by_production_region"
)

# 6) Meat & fish — by consumption region
p6 <- make_comp_plot(
  data = df,
  sector_name = "Meat & fish",
  group_var = "consumption_country",
  palette = PAL_REGION,
  fname = "fig5_meat_trade_by_consumption_region"
)

cat("\nDone! 6 figures saved.\n")