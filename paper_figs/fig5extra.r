##############################################################################
# Fig 5 composition plots (100% stacked bar)
# 6 figures:
# Fibre & Textile: by crop / production region / consumption region
# Meat & fish:     by crop / production region / consumption region
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
# HELPER: build one 100% stacked bar chart
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

  # Order legend / stacking by overall total contribution
  group_order <- plot_df %>%
    group_by(group) %>%
    summarise(total = sum(value), .groups = "drop") %>%
    arrange(desc(total)) %>%
    pull(group)

  plot_df$group <- factor(plot_df$group, levels = group_order)

  pal_use <- complete_palette(group_order, palette)

  # legend title
  fill_title <- dplyr::case_when(
    group_var == "crop" ~ "Crop",
    group_var == "production_country" ~ "Production region",
    group_var == "consumption_country" ~ "Consumption region",
    TRUE ~ group_var
  )

  p <- ggplot(plot_df, aes(x = factor(year), y = value, fill = group)) +
    geom_col(
      position = "fill",
      width = 0.95,
      colour = "#D9D9D9",
      linewidth = 0.20
    ) +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 1),
      expand = c(0, 0)
    ) +
    scale_fill_manual(values = pal_use, drop = FALSE, name = fill_title) +
    labs(
      x = NULL,
      y = "% of Total Value",
      title = NULL     # 不显示标题
    ) +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(colour = "#E6E6E6", linewidth = 0.4),
      axis.text.x = element_text(size = 10, colour = "black"),
      axis.text.y = element_text(size = 11, colour = "black"),
      axis.title.y = element_text(size = 12, colour = "black"),
      plot.title = element_blank(),   # 再保险一次：彻底去掉 title
      legend.title = element_text(size = 11, colour = "black"),
      legend.text = element_text(size = 10, colour = "black"),
      legend.position = ifelse(show_legend, "right", "none"),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
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