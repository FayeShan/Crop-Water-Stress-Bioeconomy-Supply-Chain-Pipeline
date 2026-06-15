##############################################################################
# Fig 5a & 5b: Horizontal stacked-bar Sankey (3 layers)
# Crop -> Production country -> Consumption country
# Trade only
##############################################################################

library(dplyr)
library(ggplot2)

source("config.R")

# ─────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────
df <- read.csv(AGG_CSV,
               stringsAsFactors = FALSE)

# Keep 2024 + total
df <- df %>%
  filter(year == 2024, impact_type == "total")

# Optional: multiply by 1e6 (does NOT affect shares / shape of Sankey)
# If you don't need it, you can comment this line out.
df$value <- df$value * 1e6

# Keep trade only
df <- df %>%
  filter(production_country != consumption_country)

# If needed, check sector names first:
# sort(unique(df$target_sector))

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
# LAYOUT + DIVIDER STYLE
# ─────────────────────────────────────────────────────────────────────────
BAR_H       <- 0.06
GAP         <- 0.30
N_BEZ       <- 50
DIVIDER_COL <- "#D8D8D8"
DIVIDER_A   <- 0.8
DIVIDER_LW  <- 0.25

# Three layers only
y_lay <- c(
  crop = 1.0,
  prod = 1.0 - BAR_H - GAP,
  cons = 1.0 - 2*(BAR_H + GAP)
)

# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────

# Compute bar segment positions in [0,1]
comp_bar <- function(vals, nms, tot) {
  sh <- vals[nms]
  sh[is.na(sh)] <- 0
  sh <- sh / tot

  data.frame(
    name  = nms,
    xmin  = c(0, head(cumsum(sh), -1)),
    xmax  = cumsum(sh),
    share = as.numeric(sh),
    stringsAsFactors = FALSE
  )
}

# Build one ribbon polygon using Bezier-like smooth interpolation
bez_rib <- function(x0l, x0r, y0b, x1l, x1r, y1t, n = N_BEZ) {
  t  <- seq(0, 1, length.out = n)

  lx <- (1-t)^3*x0l + 3*(1-t)^2*t*x0l + 3*(1-t)*t^2*x1l + t^3*x1l
  rx <- (1-t)^3*x0r + 3*(1-t)^2*t*x0r + 3*(1-t)*t^2*x1r + t^3*x1r

  cy <- (1-t)^3*y0b +
    3*(1-t)^2*t*(y0b - GAP*0.4) +
    3*(1-t)*t^2*(y1t + GAP*0.4) +
    t^3*y1t

  data.frame(
    x = c(lx, rev(rx)),
    y = c(cy, rev(cy))
  )
}

# Create ribbons between two layers
mk_ribs <- function(fl, bs, bt, sc, tc, ys, yt, cb, pal, mf,
                    sort_by = c("target", "source")) {
  sort_by <- match.arg(sort_by)

  s_ord <- setNames(seq_len(nrow(bs)), bs$name)
  t_ord <- setNames(seq_len(nrow(bt)), bt$name)

  fl$s_rank <- s_ord[fl[[sc]]]
  fl$t_rank <- t_ord[fl[[tc]]]

  fl <- if (sort_by == "target") {
    fl[order(fl$s_rank, fl$t_rank), ]
  } else {
    fl[order(fl$t_rank, fl$s_rank), ]
  }

  cur_s <- setNames(bs$xmin, bs$name)
  cur_t <- setNames(bt$xmin, bt$name)

  tot <- sum(fl$value)
  ribs <- list()
  cols <- c()

  for (i in seq_len(nrow(fl))) {
    s  <- fl[[sc]][i]
    tt <- fl[[tc]][i]
    v  <- fl$value[i]
    w  <- v / tot

    if (w < mf) next

    x0l <- cur_s[s]
    x0r <- x0l + w
    cur_s[s] <- x0r

    x1l <- cur_t[tt]
    x1r <- x1l + w
    cur_t[tt] <- x1r

    po <- bez_rib(x0l, x0r, ys - BAR_H, x1l, x1r, yt)
    po$id <- i

    ribs[[length(ribs) + 1]] <- po

    k <- if (cb == "source") s else tt
    cols <- c(cols, ifelse(k %in% names(pal), pal[k], "#CCCCCC"))
  }

  list(ribbons = ribs, colors = cols)
}

# Draw ribbons
dr_ribs <- function(p, rl, al = 0.60) {
  for (i in seq_along(rl$ribbons)) {
    p <- p + geom_polygon(
      data = rl$ribbons[[i]],
      aes(x = x, y = y, group = id),
      fill = rl$colors[i], alpha = al, colour = NA
    )
  }
  p
}

# Abbreviate labels for small segments
abbrev <- function(nm, sh) {
  if (sh >= 0.06) return(nm)

  nm <- gsub("Other Middle East & North Africa", "O.Middle East\n& N.Afr.", nm)
  nm <- gsub("Other Asia-Pacific",               "Other\nAsia-Pac",        nm)
  nm <- gsub("Other America",                    "Other\nAmerica",         nm)
  nm <- gsub("Sub-Saharan Africa",               "Sub-Saharan\nAfrica",    nm)
  nm <- gsub("United States",                    "United\nStates",         nm)
  nm <- gsub("Other Cereals",                    "Other\nCereals",         nm)
  nm <- gsub("Other Crops",                      "Other\nCrops",           nm)

  nm
}

# Choose white or dark text based on luminance
label_color <- function(hex) {
  rgb <- col2rgb(hex) / 255
  lum <- 0.2126 * rgb[1] + 0.7152 * rgb[2] + 0.0722 * rgb[3]
  if (lum > 0.70) "#333333" else "white"
}

# Rough estimate: whether label can fit inside the bar segment
# box_w / box_h are in plot data units
# size_mm is annotate(..., size = ...)
label_fits <- function(label, box_w, box_h, size_mm,
                       char_w = 0.0105,
                       line_h = 0.030,
                       width_pad = 0.92,
                       height_pad = 0.88) {
  lines <- strsplit(label, "\n")[[1]]
  max_chars <- max(nchar(lines))
  n_lines   <- length(lines)

  # Approximate required width/height in data units
  est_w <- max_chars * char_w * (size_mm / 3.5)
  est_h <- n_lines   * line_h * (size_mm / 3.5)

  (est_w <= box_w * width_pad) && (est_h <= box_h * height_pad)
}

# Draw one horizontal stacked bar with labels
# lsz controls label size; larger than before
dr_bar <- function(p, bd, yt, pal, lab = TRUE, lsz = 3.8) {
  for (i in 1:nrow(bd)) {
    nm <- bd$name[i]
    co <- ifelse(nm %in% names(pal), pal[nm], "#AAAAAA")

    # segment fill
    p <- p + annotate("rect",
                      xmin = bd$xmin[i], xmax = bd$xmax[i],
                      ymin = yt - BAR_H, ymax = yt,
                      fill = co, colour = NA)

    # label (only if it fits)
    if (lab) {
      mx <- (bd$xmin[i] + bd$xmax[i]) / 2
      my <- yt - BAR_H / 2

      sn <- abbrev(nm, bd$share[i])
      lt <- paste0(sn, "\n", round(bd$share[i] * 100, 1), "%")

      box_w <- bd$xmax[i] - bd$xmin[i]
      box_h <- BAR_H

      if (bd$share[i] > 0.01 &&
          label_fits(lt, box_w, box_h, lsz)) {
        p <- p + annotate("text",
                          x = mx, y = my, label = lt,
                          size = lsz, fontface = "bold",
                          colour = label_color(co),
                          lineheight = 0.85)
      }
    }
  }

  # inner dividers
  inner_x <- bd$xmax[-nrow(bd)]
  for (xv in inner_x) {
    p <- p + annotate("segment",
                      x = xv, xend = xv,
                      y = yt - BAR_H, yend = yt,
                      colour = DIVIDER_COL, alpha = DIVIDER_A,
                      linewidth = DIVIDER_LW)
  }

  # outer outline
  p <- p + annotate("rect",
                    xmin = min(bd$xmin), xmax = max(bd$xmax),
                    ymin = yt - BAR_H, ymax = yt,
                    fill = NA, colour = DIVIDER_COL,
                    alpha = DIVIDER_A * 0.7, linewidth = DIVIDER_LW)

  p
}

# ─────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION FOR 3-LAYER FIGURE
# ─────────────────────────────────────────────────────────────────────────
build_fig3 <- function(data, lab, mf, fname,
                       outdir = file.path(OUTPUT_DIR, "sankeys")) {

  tv <- sum(data$value)

  # Flow tables
  f1 <- data %>%
    group_by(crop, production_country) %>%
    summarise(value = sum(value), .groups = "drop")

  f2 <- data %>%
    group_by(production_country, consumption_country) %>%
    summarise(value = sum(value), .groups = "drop")

  # Totals for bars
  cr_tot <- data %>%
    group_by(crop) %>%
    summarise(v = sum(value), .groups = "drop")

  pr_tot <- data %>%
    group_by(production_country) %>%
    summarise(vp = sum(value), .groups = "drop")

  cn_tot <- data %>%
    group_by(consumption_country) %>%
    summarise(vc = sum(value), .groups = "drop")

  # Unified order for production / consumption countries
  all_country <- full_join(
    pr_tot, cn_tot,
    by = c("production_country" = "consumption_country")
  ) %>%
    rename(country = production_country) %>%
    mutate(
      vp = ifelse(is.na(vp), 0, vp),
      vc = ifelse(is.na(vc), 0, vc),
      v_total = vp + vc
    ) %>%
    arrange(desc(v_total))

  country_order <- all_country$country
  prod_order <- country_order[country_order %in% pr_tot$production_country]
  cons_order <- country_order[country_order %in% cn_tot$consumption_country]

  cr <- cr_tot %>% arrange(desc(v))

  bc <- comp_bar(setNames(cr$v, cr$crop), cr$crop, tv)
  bp <- comp_bar(setNames(pr_tot$vp, pr_tot$production_country), prod_order, tv)
  bn <- comp_bar(setNames(cn_tot$vc, cn_tot$consumption_country), cons_order, tv)

  # Ribbons
  r1 <- mk_ribs(
    f1, bc, bp,
    "crop", "production_country",
    y_lay["crop"], y_lay["prod"],
    "source", PAL_CROP, mf,
    sort_by = "target"
  )

  r2 <- mk_ribs(
    f2, bp, bn,
    "production_country", "consumption_country",
    y_lay["prod"], y_lay["cons"],
    "source", PAL_REGION, mf,
    sort_by = "target"
  )

  # Plot
  p <- ggplot() +
    theme_void() +
    coord_cartesian(
      xlim = c(0, 1),
      ylim = c(y_lay["cons"] - BAR_H - 0.02,
               y_lay["crop"] + BAR_H + 0.01)
    )

  p <- dr_ribs(p, r1)
  p <- dr_ribs(p, r2)

  # Larger labels, auto hidden if they do not fit
  p <- dr_bar(p, bc, y_lay["crop"], PAL_CROP,   lab, lsz = 3.0)
  p <- dr_bar(p, bp, y_lay["prod"], PAL_REGION, lab, lsz = 3.0)
  p <- dr_bar(p, bn, y_lay["cons"], PAL_REGION, lab, lsz = 3.0)

  p <- p + theme(
    plot.margin      = margin(5, 5, 5, 5),
    plot.background  = element_rect(fill = "transparent", colour = NA),
    panel.background = element_rect(fill = "transparent", colour = NA)
  )

  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

  ggsave(file.path(outdir, paste0(fname, ".pdf")),
         p, width = 16, height = 8, units = "in",
         bg = "transparent")

  ggsave(file.path(outdir, paste0(fname, ".png")),
         p, width = 16, height = 8, units = "in", dpi = 300,
         bg = "transparent")

  cat("Saved:", fname, "\n")
}

# ─────────────────────────────────────────────────────────────────────────
# GENERATE FIG 5A: Fibre & Textile (trade only)
# ─────────────────────────────────────────────────────────────────────────
df_fibre <- df %>%
  filter(target_sector == "Fibre & Textile")

build_fig3(df_fibre, lab = TRUE,  mf = 0.001, "fig5a_fibre_trade_labeled")
build_fig3(df_fibre, lab = FALSE, mf = 0.001, "fig5a_fibre_trade_nolabel")

# ─────────────────────────────────────────────────────────────────────────
# GENERATE FIG 5B: Meat & fish (trade only)
# ─────────────────────────────────────────────────────────────────────────
df_meat <- df %>%
  filter(target_sector == "Meat & fish")

build_fig3(df_meat, lab = TRUE,  mf = 0.001, "fig5b_meat_trade_labeled")
build_fig3(df_meat, lab = FALSE, mf = 0.001, "fig5b_meat_trade_nolabel")

cat("\nDone! 4 versions saved.\n")

##############################################################################
# Save label-value table for Fig 5 Sankey
##############################################################################

save_fig5_label_table <- function(data,
                                  sector_name,
                                  fig_name,
                                  outdir = file.path(OUTPUT_DIR, "sankeys"),
                                  lsz = 3.8) {

  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

  df_sub <- data %>%
    filter(target_sector == sector_name)

  tv <- sum(df_sub$value)

  # ---- helper: summarize one layer ----
  make_layer_table <- function(df_sub, group_var, layer_name, palette) {

    out <- df_sub %>%
      group_by(.data[[group_var]]) %>%
      summarise(value = sum(value), .groups = "drop") %>%
      rename(label = .data[[group_var]]) %>%
      arrange(desc(value)) %>%
      mutate(
        figure = fig_name,
        target_sector = sector_name,
        layer = layer_name,
        share = value / tv,
        share_percent = share * 100,
        abbreviated_label = mapply(abbrev, label, share),
        label_text = paste0(abbreviated_label, "\n", round(share_percent, 1), "%")
      )

    # match the same "label shown or hidden" rule as dr_bar()
    out <- out %>%
      rowwise() %>%
      mutate(
        color = ifelse(label %in% names(palette), palette[label], "#AAAAAA"),
        box_w = share,
        box_h = BAR_H,
        shown_by_rule = share > 0.01 &&
          label_fits(label_text, box_w, box_h, lsz)
      ) %>%
      ungroup()

    out %>%
      select(
        figure, target_sector, layer, label,
        value, share, share_percent,
        abbreviated_label, label_text,
        shown_by_rule
      )
  }

  crop_tab <- make_layer_table(
    df_sub = df_sub,
    group_var = "crop",
    layer_name = "Crop",
    palette = PAL_CROP
  )

  prod_tab <- make_layer_table(
    df_sub = df_sub,
    group_var = "production_country",
    layer_name = "Production country",
    palette = PAL_REGION
  )

  cons_tab <- make_layer_table(
    df_sub = df_sub,
    group_var = "consumption_country",
    layer_name = "Consumption country",
    palette = PAL_REGION
  )

  bind_rows(crop_tab, prod_tab, cons_tab)
}

# ─────────────────────────────────────────────────────────────────────────
# Generate and save one combined CSV for Fig 5a + Fig 5b
# 注意：这里用的 df 应该已经是：
# year == 2024, impact_type == "total", trade only
# 也就是你前面代码里已经处理过的 df
# ─────────────────────────────────────────────────────────────────────────

label_table_fig5 <- bind_rows(
  save_fig5_label_table(
    data = df,
    sector_name = "Fibre & Textile",
    fig_name = "fig5a_fibre_trade",
    lsz = 3.8
  ),
  save_fig5_label_table(
    data = df,
    sector_name = "Meat & fish",
    fig_name = "fig5b_meat_trade",
    lsz = 3.8
  )
)

write.csv(
  label_table_fig5,
  file.path(OUTPUT_DIR, "sankeys", "fig5_sankey_label_values.csv"),
  row.names = FALSE
)

cat("Saved label table: fig5_sankey_label_values.csv\n")