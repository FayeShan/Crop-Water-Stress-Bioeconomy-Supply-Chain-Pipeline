##############################################################################
# Fig 4a & 4b: Horizontal stacked-bar Sankey 
##############################################################################

library(dplyr)
library(ggplot2)

source("config.R")

df <- read.csv(AGG_CSV,
               stringsAsFactors = FALSE)
df <- df %>% filter(year == 2024, impact_type == "total")
df$value <- df$value * 1e6
total_val <- sum(df$value)

# ─────────────────────────────────────────────────────────────────────────
# PALETTES
# ─────────────────────────────────────────────────────────────────────────

# REGION  — United States / Iran / Other MENA were too similar; now clearly
# separated (US = deep indigo, Iran = green, O.MENA = warm amber).
PAL_REGION <- c(
    "India"                              = "#1F4E79",  # deep navy
    "China"                              = "#C1272D",  # signature red (biggest block)
    "Pakistan"                           = "#2E86AB",  # mid blue
    "Other Asia-Pacific"                 = "#8E6E3C",  
    "Other Middle East & North Africa"   = "#E8A33D",  # warm amber   (was teal)
    "Egypt"                              = "#71a836",  # rose-red
    "Iran"                               = "#4B9F7E",  # green        (was teal)
    "Europe"                             = "#6A4C93",  # olive-brown
    "Other America"                      = "#7FB3D5",  # soft sky blue
    "Sub-Saharan Africa"                 = "#B84A8A",  # magenta
    "United States"                      = "#2C3E75"   # deep indigo  (was teal-green)
)

# # CROP — each color evokes the crop itself
# PAL_CROP <- c(
#     "Wheat"         = "#D4A24C",   # golden straw
#     "Rice"          = "#F5ECD2",   # rice white / cream
#     "Maize"         = "#F2C14E",   # corn yellow
#     "Other Cereals" = "#A88B4C",   # grain brown
#     "Fibre Crops"   = "#F0F0EA",   # cotton off-white
#     "Oil Seeds"     = "#6B8E23",   # olive / soybean green
#     "Sugar Crops"   = "#E8D4A0",   # raw cane sugar
#     "Fruits & nuts" = "#C73E3A",   # berry red
#     "Vegetables"    = "#3E7C47",   # leaf green
#     "Livestock"     = "#8B4513",   # saddle brown
#     "Other Crops"   = "#B0A58E"    # muted tan
# )

# # SECTOR — each color evokes the product
# PAL_SECTOR <- c(
#     "Cereals"                  = "#C49A3F",  # bread / wheat tan
#     "Rice"                     = "#E8DCAE",  # rice hull
#     "Fibre & Textile"          = "#5D4E8C",  # indigo dye
#     "Oil products"             = "#DAAE4B",  # olive-oil gold
#     "Sugar & chocolate"        = "#5C3317",  # cocoa brown
#     "Fruits & nuts"            = "#D96A8F",  # fruit pink
#     "Vegetables"               = "#4A8B3A",  # fresh green
#     "Other agri-food products" = "#8A7F6E",  # neutral taupe
#     "Meat & fish"              = "#7A1E2B",  # raw meat red
#     "Dairy"                    = "#F4EBD8",  # milk cream
#     "Beverages"                = "#3B2B1F",  # coffee black
#     "Biochemicals"             = "#2E7D7A"   # lab teal
# )

# PAL_REGION <- c(
#     "India"                              = "#00A6A6",  # strong teal
#     "China"                              = "#5147D9",  # vivid blue-violet
#     "Pakistan"                           = "#008FD3",  # bright blue
#     "Other Asia-Pacific"                 = "#7B61FF",  # purple-blue
#     "Other Middle East & North Africa"   = "#C06CFF",  # violet
#     "Egypt"                              = "#2D6CDF",  # royal blue
#     "Iran"                               = "#00B894",  # turquoise green
#     "Europe"                             = "#6C5CE7",  # deep lavender
#     "Other America"                      = "#00C2D1",  # cyan-teal
#     "Sub-Saharan Africa"                 = "#9B5DE5",  # purple
#     "United States"                      = "#003F88"   # dark blue
# )

PAL_CROP <- c(
    "Wheat"         = "#E76F00",  # burnt orange
    "Rice"          = "#D9C75F",  # muted rice yellow
    "Maize"         = "#FFD23F",  # corn yellow
    "Other Cereals" = "#8A9A32",  # olive green
    "Fibre Crops"   = "#E63946",  # cotton/fibre red
    "Oil Seeds"     = "#4F8A10",  # deep oilseed green
    "Sugar Crops"   = "#C9A227",  # sugarcane gold
    "Fruits & nuts" = "#D1495B",  # fruit red
    "Vegetables"    = "#2A9D3F",  # vegetable green
    "Livestock"     = "#8B5E34",  # livestock brown
    "Other Crops"   = "#B5651D"   # earthy orange-brown
)

PAL_SECTOR <- c(
    "Cereals"                  = "#F9C80E",  # bright yellow
    "Rice"                     = "#F86624",  # vivid orange-red
    "Fibre & Textile"          = "#ae138c",  # bright pink
    "Oil products"             = "#277DA1",  # petroleum blue
    "Sugar & chocolate"        = "#6D4C41",  # chocolate brown
    "Fruits & nuts"            = "#f388b1",  # fruit pink
    "Vegetables"               = "#70E000",  # bright green
    "Other agri-food products" = "#F8961E",  # orange
    "Meat & fish"              = "#B5172F",  # meat red
    "Dairy"                    = "#A7C7E7",  # milk blue
    "Beverages"                = "#3A0CA3",  # dark beverage purple
    "Biochemicals"             = "#00B4D8"   # chemical cyan
)


# ─────────────────────────────────────────────────────────────────────────
# LAYOUT  +  DIVIDER STYLE
# ─────────────────────────────────────────────────────────────────────────
BAR_H       <- 0.06
GAP         <- 0.30
N_BEZ       <- 50
DIVIDER_COL <- "#D8D8D8"   # light grey (replaces opaque white between segments)
DIVIDER_A   <- 0.8         # translucent
DIVIDER_LW  <- 0.25

y_lay <- c(crop = 1.0,
           prod = 1.0 - BAR_H - GAP,
           cons = 1.0 - 2*(BAR_H + GAP),
           sect = 1.0 - 3*(BAR_H + GAP))

# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────

comp_bar <- function(vals, nms, tot) {
    sh <- vals[nms]; sh[is.na(sh)] <- 0
    sh <- sh / tot
    data.frame(name  = nms,
               xmin  = c(0, head(cumsum(sh), -1)),
               xmax  = cumsum(sh),
               share = as.numeric(sh),
               stringsAsFactors = FALSE)
}

bez_rib <- function(x0l, x0r, y0b, x1l, x1r, y1t, n = N_BEZ) {
    t  <- seq(0, 1, length.out = n)
    lx <- (1-t)^3*x0l + 3*(1-t)^2*t*x0l + 3*(1-t)*t^2*x1l + t^3*x1l
    rx <- (1-t)^3*x0r + 3*(1-t)^2*t*x0r + 3*(1-t)*t^2*x1r + t^3*x1r
    cy <- (1-t)^3*y0b + 3*(1-t)^2*t*(y0b-GAP*0.4) +
          3*(1-t)*t^2*(y1t+GAP*0.4) + t^3*y1t
    data.frame(x = c(lx, rev(rx)), y = c(cy, rev(cy)))
}

mk_ribs <- function(fl, bs, bt, sc, tc, ys, yt, cb, pal, mf,
                    sort_by = c("target", "source")) {
    sort_by <- match.arg(sort_by)
    s_ord <- setNames(seq_len(nrow(bs)), bs$name)
    t_ord <- setNames(seq_len(nrow(bt)), bt$name)
    fl$s_rank <- s_ord[fl[[sc]]]
    fl$t_rank <- t_ord[fl[[tc]]]
    fl <- if (sort_by == "target")
              fl[order(fl$s_rank, fl$t_rank), ]
          else
              fl[order(fl$t_rank, fl$s_rank), ]

    cur_s <- setNames(bs$xmin, bs$name)
    cur_t <- setNames(bt$xmin, bt$name)
    tot <- sum(fl$value)
    ribs <- list(); cols <- c()
    for (i in seq_len(nrow(fl))) {
        s  <- fl[[sc]][i]; tt <- fl[[tc]][i]
        v  <- fl$value[i]; w <- v / tot
        if (w < mf) next
        x0l <- cur_s[s]; x0r <- x0l + w; cur_s[s] <- x0r
        x1l <- cur_t[tt]; x1r <- x1l + w; cur_t[tt] <- x1r
        po <- bez_rib(x0l, x0r, ys - BAR_H, x1l, x1r, yt); po$id <- i
        ribs[[length(ribs) + 1]] <- po
        k <- if (cb == "source") s else tt
        cols <- c(cols, ifelse(k %in% names(pal), pal[k], "#CCCCCC"))
    }
    list(ribbons = ribs, colors = cols)
}

dr_ribs <- function(p, rl, al = 0.60) {
    for (i in seq_along(rl$ribbons))
        p <- p + geom_polygon(data = rl$ribbons[[i]],
                              aes(x = x, y = y, group = id),
                              fill = rl$colors[i], alpha = al, colour = NA)
    p
}

abbrev <- function(nm, sh) {
    if (sh >= 0.06) return(nm)
    nm <- gsub("Other Middle East & North Africa", "O.Middle East\n& N.Afr.", nm)
    nm <- gsub("Other Asia-Pacific",          "Other\nAsia-Pac",     nm)
    nm <- gsub("Other America",               "Other\nAmerica",      nm)
    nm <- gsub("Sub-Saharan Africa",          "Sub-Saharan\nAfrica", nm)
    nm <- gsub("United States",               "United\nStates",      nm)
    nm <- gsub("Other agri-food products",    "O.agri-food\nproducts", nm)
    nm <- gsub("Other Cereals",               "Other\nCereals",      nm)
    nm <- gsub("Other Crops",                 "Other\nCrops",        nm)
    nm <- gsub("Sugar & chocolate",           "Sugar &\nchocolate",  nm)
    nm <- gsub("Fibre & Textile",             "Fibre &\nTextile",    nm)
    nm <- gsub("Fruits & nuts",               "Fruits &\nnuts",      nm)
    nm <- gsub("Oil products",                "Oil\nproducts",       nm)
    nm
}

# Decide whether a label on a colored bar should be dark or white,
# based on fill luminance.  Important now that several fills are very pale
# (Rice, Fibre Crops, Sugar Crops, Dairy).
label_color <- function(hex) {
    rgb <- col2rgb(hex) / 255
    # Relative luminance (sRGB)
    lum <- 0.2126*rgb[1] + 0.7152*rgb[2] + 0.0722*rgb[3]
    if (lum > 0.70) "#333333" else "white"
}

dr_bar <- function(p, bd, yt, pal, lab = TRUE, lsz = 2.5) {
    for (i in 1:nrow(bd)) {
        nm <- bd$name[i]
        co <- ifelse(nm %in% names(pal), pal[nm], "#AAAAAA")
        # fill (no stroke) — dividers drawn separately so they can be
        # translucent light grey instead of opaque white.
        p <- p + annotate("rect",
                          xmin = bd$xmin[i], xmax = bd$xmax[i],
                          ymin = yt - BAR_H, ymax = yt,
                          fill = co, colour = NA)
        if (lab && bd$share[i] > 0.025) {
            mx <- (bd$xmin[i] + bd$xmax[i]) / 2
            my <- yt - BAR_H / 2
            sn <- abbrev(nm, bd$share[i])
            lt <- paste0(sn, "\n", round(bd$share[i] * 100, 1), "%")
            p  <- p + annotate("text", x = mx, y = my, label = lt,
                               size = lsz, fontface = "bold",
                               colour = label_color(co), lineheight = 0.85)
        }
    }
    # translucent light-grey dividers between segments + around the bar
    inner_x <- bd$xmax[-nrow(bd)]
    for (xv in inner_x) {
        p <- p + annotate("segment",
                          x = xv, xend = xv,
                          y = yt - BAR_H, yend = yt,
                          colour = DIVIDER_COL, alpha = DIVIDER_A,
                          linewidth = DIVIDER_LW)
    }
    # outer outline (also light grey, half strength)
    p <- p + annotate("rect",
                      xmin = min(bd$xmin), xmax = max(bd$xmax),
                      ymin = yt - BAR_H, ymax = yt,
                      fill = NA, colour = DIVIDER_COL,
                      alpha = DIVIDER_A * 0.7, linewidth = DIVIDER_LW)
    p
}

# ─────────────────────────────────────────────────────────────────────────
# MASTER
# ─────────────────────────────────────────────────────────────────────────
build_fig <- function(data, lab, mf, fname) {
    tv <- sum(data$value)

    f1 <- data %>% group_by(crop, production_country) %>%
            summarise(value = sum(value), .groups = "drop")
    f2 <- data %>% group_by(production_country, consumption_country) %>%
            summarise(value = sum(value), .groups = "drop")
    f3 <- data %>% group_by(consumption_country, target_sector) %>%
            summarise(value = sum(value), .groups = "drop")

    cr_tot <- data %>% group_by(crop) %>% summarise(v = sum(value))
    pr_tot <- data %>% group_by(production_country)  %>% summarise(vp = sum(value))
    cn_tot <- data %>% group_by(consumption_country) %>% summarise(vc = sum(value))
    sc_tot <- data %>% group_by(target_sector) %>% summarise(v = sum(value))

    # Unified country order (prod + cons combined) — so the same country sits
    # at (near-)identical x on both layers.
    all_country <- full_join(pr_tot, cn_tot,
                             by = c("production_country" = "consumption_country")) %>%
                   rename(country = production_country) %>%
                   mutate(vp = ifelse(is.na(vp), 0, vp),
                          vc = ifelse(is.na(vc), 0, vc),
                          v_total = vp + vc) %>%
                   arrange(desc(v_total))
    country_order <- all_country$country
    prod_order <- country_order[country_order %in% pr_tot$production_country]
    cons_order <- country_order[country_order %in% cn_tot$consumption_country]

    cr <- cr_tot %>% arrange(desc(v))
    sc <- sc_tot %>% arrange(desc(v))

    bc <- comp_bar(setNames(cr$v, cr$crop), cr$crop, tv)
    bp <- comp_bar(setNames(pr_tot$vp, pr_tot$production_country),  prod_order, tv)
    bn <- comp_bar(setNames(cn_tot$vc, cn_tot$consumption_country), cons_order, tv)
    bs <- comp_bar(setNames(sc$v, sc$target_sector), sc$target_sector, tv)

    r1 <- mk_ribs(f1, bc, bp, "crop", "production_country",
                  y_lay["crop"], y_lay["prod"], "source", PAL_CROP, mf,
                  sort_by = "target")
    r2 <- mk_ribs(f2, bp, bn, "production_country", "consumption_country",
                  y_lay["prod"], y_lay["cons"], "source", PAL_REGION, mf,
                  sort_by = "target")
    r3 <- mk_ribs(f3, bn, bs, "consumption_country", "target_sector",
                  y_lay["cons"], y_lay["sect"], "target", PAL_SECTOR, mf,
                  sort_by = "source")

    p <- ggplot() + theme_void() +
        coord_cartesian(xlim = c(0, 1),
                        ylim = c(y_lay["sect"] - BAR_H - 0.02,
                                 y_lay["crop"] + BAR_H + 0.01))

    p <- dr_ribs(p, r1); p <- dr_ribs(p, r2); p <- dr_ribs(p, r3)
    p <- dr_bar(p, bc, y_lay["crop"], PAL_CROP,   lab)
    p <- dr_bar(p, bp, y_lay["prod"], PAL_REGION, lab)
    p <- dr_bar(p, bn, y_lay["cons"], PAL_REGION, lab)
    p <- dr_bar(p, bs, y_lay["sect"], PAL_SECTOR, lab)

    p <- p + theme(plot.margin      = margin(5, 5, 5, 5),
               plot.background  = element_rect(fill = "transparent", colour = NA),
               panel.background = element_rect(fill = "transparent", colour = NA))

    outdir <- file.path(OUTPUT_DIR, "sankeys")
    dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
    ggsave(file.path(outdir, paste0(fname, ".pdf")), p,
       width = 16, height = 10, units = "in",
       bg = "transparent")

    ggsave(file.path(outdir, paste0(fname, ".png")), p,
       width = 16, height = 10, units = "in", dpi = 300,
       bg = "transparent")
    cat("Saved:", fname, "\n")
}

# ═════════════════════════════════════════════════════════════════════════
# GENERATE
# ═════════════════════════════════════════════════════════════════════════

# 4a global
build_fig(df, lab = TRUE,  mf = 0.002, "fig4a_global_labeled")
build_fig(df, lab = FALSE, mf = 0.002, "fig4a_global_nolabel")

# 4b trade only
df_t <- df %>% filter(production_country != consumption_country)
build_fig(df_t, lab = TRUE,  mf = 0.001, "fig4b_trade_labeled")
build_fig(df_t, lab = FALSE, mf = 0.001, "fig4b_trade_nolabel")

cat("\nDone! 4 versions saved.\n")