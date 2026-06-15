"""Generate the 175-FAO-crops -> 27-CWS-crops -> 14-GLORIA-sectors mapping CSV (the single source of truth for the pipeline).

``sectors_14`` drops "Seeds and plant propagation" because the CWS
water-stress data doesn't cover it.
"""

from pathlib import Path
from typing import Optional
import logging

import pandas as pd


# 14 GLORIA crop sectors
sectors_14 = [
    "Growing wheat",
    "Growing maize",
    "Growing cereals n.e.c",
    "Growing leguminous crops and oil seeds",
    "Growing rice",
    "Growing vegetables, roots, tubers",
    "Growing sugar beet and cane",
    "Growing tobacco",
    "Growing fibre crops",
    "Growing crops n.e.c.",
    "Growing grapes",
    "Growing fruits and nuts",
    "Growing beverage crops (coffee, tea etc)",
    "Growing spices, aromatic, drug and pharmaceutical crops",
]

# 27 CWS crop categories (abbreviation -> display name)
crops_27 = {
    "wh": "Wheat",
    "ri1": "Rice (main season)",
    "ri2": "Rice (second season)",
    "mai": "Maize (corn)",
    "sor": "Sorghum",
    "mil": "Millet",
    "bar": "Barley",
    "soy": "Soya beans",
    "nut": "Groundnuts",
    "sun": "Sunflower seed",
    "rap": "Rape or colza seed",
    "bea": "Beans",
    "ckp": "Chick peas",
    "cwp": "Cow peas",
    "pot": "Potatoes",
    "cas": "Cassava",
    "vgt": "Vegetables",
    "sgc": "Sugar cane",
    "sgb": "Sugar beet",
    "cot": "Seed cotton",
    "cof": "Coffee",
    "coc": "Cocoa beans",
    "plm": "Oil palm fruit",
    "aff": "Alfalfa",
    "oac": "Other annual crops",
    "pdc": "Perennial deciduous crops",
    "pec": "Perennial evergreen crops",
}


# Complete FAO 175-crop -> (crop_27 category, sector_14) mapping
crop_mapping = {
    "Wheat": ("wh", "Growing wheat"),

    "Rice": ("ri1", "Growing rice"),

    "Maize (corn)": ("mai", "Growing maize"),
    "Green corn (maize)": ("mai", "Growing maize"),

    "Sorghum": ("sor", "Growing cereals n.e.c"),

    "Millet": ("mil", "Growing cereals n.e.c"),
    "Fonio": ("mil", "Growing cereals n.e.c"),

    "Barley": ("bar", "Growing cereals n.e.c"),

    # ===== OTHER CEREALS (-> oac, cereals n.e.c) =====
    "Cereals n.e.c.": ("oac", "Growing cereals n.e.c"),
    "Rye": ("oac", "Growing cereals n.e.c"),
    "Triticale": ("oac", "Growing cereals n.e.c"),
    "Buckwheat": ("oac", "Growing cereals n.e.c"),
    "Mixed grain": ("oac", "Growing cereals n.e.c"),
    "Oats": ("oac", "Growing cereals n.e.c"),
    "Canary seed": ("oac", "Growing cereals n.e.c"),
    "Quinoa": ("oac", "Growing cereals n.e.c"),
    
    "Soya beans": ("soy", "Growing leguminous crops and oil seeds"),

    "Groundnuts, excluding shelled": ("nut", "Growing leguminous crops and oil seeds"),

    "Sunflower seed": ("sun", "Growing leguminous crops and oil seeds"),

    "Rape or colza seed": ("rap", "Growing leguminous crops and oil seeds"),

    "Beans, dry": ("bea", "Growing leguminous crops and oil seeds"),
    "Broad beans and horse beans, dry": ("bea", "Growing leguminous crops and oil seeds"),
    "Bambara beans, dry": ("bea", "Growing leguminous crops and oil seeds"),

    "Chick peas, dry": ("ckp", "Growing leguminous crops and oil seeds"),

    "Cow peas, dry": ("cwp", "Growing leguminous crops and oil seeds"),
    "Pigeon peas, dry": ("cwp", "Growing leguminous crops and oil seeds"),

    # ===== OTHER LEGUMES/OILSEEDS (-> oac, leguminous) =====
    "Peas, dry": ("oac", "Growing leguminous crops and oil seeds"),
    "Lentils, dry": ("oac", "Growing leguminous crops and oil seeds"),
    "Lupins": ("oac", "Growing leguminous crops and oil seeds"),
    "Vetches": ("oac", "Growing leguminous crops and oil seeds"),
    "Other pulses n.e.c.": ("oac", "Growing leguminous crops and oil seeds"),
    "Linseed": ("oac", "Growing leguminous crops and oil seeds"),
    "Melonseed": ("oac", "Growing leguminous crops and oil seeds"),
    "Other oil seeds, n.e.c.": ("oac", "Growing leguminous crops and oil seeds"),
    "Castor oil seeds": ("oac", "Growing leguminous crops and oil seeds"),
    "Mustard seed": ("oac", "Growing leguminous crops and oil seeds"),
    "Safflower seed": ("oac", "Growing leguminous crops and oil seeds"),
    "Sesame seed": ("oac", "Growing leguminous crops and oil seeds"),
    "Tallowtree seeds": ("oac", "Growing leguminous crops and oil seeds"),
    "Kapok fruit": ("oac", "Growing leguminous crops and oil seeds"),
    "Tung nuts": ("oac", "Growing leguminous crops and oil seeds"),
    "Poppy seed": ("oac", "Growing leguminous crops and oil seeds"),
    "Hempseed": ("oac", "Growing leguminous crops and oil seeds"),
    "Jojoba seeds": ("oac", "Growing leguminous crops and oil seeds"),

    "Potatoes": ("pot", "Growing vegetables, roots, tubers"),
    "Sweet potatoes": ("pot", "Growing vegetables, roots, tubers"),

    "Cassava, fresh": ("cas", "Growing vegetables, roots, tubers"),

    "Tomatoes": ("vgt", "Growing vegetables, roots, tubers"),
    "Onions and shallots, dry (excluding dehydrated)": ("vgt", "Growing vegetables, roots, tubers"),
    "Onions and shallots, green": ("vgt", "Growing vegetables, roots, tubers"),
    "Chillies and peppers, green (Capsicum spp. and Pimenta spp.)": ("vgt", "Growing vegetables, roots, tubers"),
    "Yams": ("vgt", "Growing vegetables, roots, tubers"),
    "Taro": ("vgt", "Growing vegetables, roots, tubers"),
    "Yautia": ("vgt", "Growing vegetables, roots, tubers"),
    "Edible roots and tubers with high starch or inulin content, n.e.c., fresh": ("vgt", "Growing vegetables, roots, tubers"),
    "Cucumbers and gherkins": ("vgt", "Growing vegetables, roots, tubers"),
    "Cabbages": ("vgt", "Growing vegetables, roots, tubers"),
    "Okra": ("vgt", "Growing vegetables, roots, tubers"),
    "Peas, green": ("vgt", "Growing vegetables, roots, tubers"),
    "Watermelons": ("vgt", "Growing vegetables, roots, tubers"),
    "Eggplants (aubergines)": ("vgt", "Growing vegetables, roots, tubers"),
    "Other beans, green": ("vgt", "Growing vegetables, roots, tubers"),
    "Green garlic": ("vgt", "Growing vegetables, roots, tubers"),
    "Asparagus": ("vgt", "Growing vegetables, roots, tubers"),
    "Pumpkins, squash and gourds": ("vgt", "Growing vegetables, roots, tubers"),
    "Cauliflowers and broccoli": ("vgt", "Growing vegetables, roots, tubers"),
    "Lettuce and chicory": ("vgt", "Growing vegetables, roots, tubers"),
    "Carrots and turnips": ("vgt", "Growing vegetables, roots, tubers"),
    "Cantaloupes and other melons": ("vgt", "Growing vegetables, roots, tubers"),
    "Spinach": ("vgt", "Growing vegetables, roots, tubers"),
    "Broad beans and horse beans, green": ("vgt", "Growing vegetables, roots, tubers"),
    "String beans": ("vgt", "Growing vegetables, roots, tubers"),
    "Leeks and other alliaceous vegetables": ("vgt", "Growing vegetables, roots, tubers"),
    "Artichokes": ("vgt", "Growing vegetables, roots, tubers"),
    "Mushrooms and truffles": ("vgt", "Growing vegetables, roots, tubers"),
    "Chicory roots": ("vgt", "Growing vegetables, roots, tubers"),
    "Other vegetables, fresh n.e.c.": ("vgt", "Growing vegetables, roots, tubers"),

    "Sugar cane": ("sgc", "Growing sugar beet and cane"),
    "Other sugar crops n.e.c.": ("sgc", "Growing sugar beet and cane"),

    "Sugar beet": ("sgb", "Growing sugar beet and cane"),

    "Seed cotton, unginned": ("cot", "Growing fibre crops"),

    # ===== OTHER FIBRE CROPS (-> oac, fibre crops) =====
    "Jute, raw or retted": ("oac", "Growing fibre crops"),
    "Other fibre crops, raw, n.e.c.": ("oac", "Growing fibre crops"),
    "Flax, processed but not spun": ("oac", "Growing fibre crops"),
    "Sisal, raw": ("oac", "Growing fibre crops"),
    "Abaca, manila hemp, raw": ("oac", "Growing fibre crops"),
    "Kenaf, and other textile bast fibres, raw or retted": ("oac", "Growing fibre crops"),
    "True hemp, raw or retted": ("oac", "Growing fibre crops"),
    "Agave fibres, raw, n.e.c.": ("oac", "Growing fibre crops"),
    "Ramie, raw or retted": ("oac", "Growing fibre crops"),

    "Coffee, green": ("cof", "Growing beverage crops (coffee, tea etc)"),

    "Cocoa beans": ("coc", "Growing beverage crops (coffee, tea etc)"),

    "Oil palm fruit": ("plm", "Growing fruits and nuts"),  # Could also be oilseeds

    "Forage and silage, alfalfa": ("aff", "Growing crops n.e.c."),
    "Clover for forage": ("aff", "Growing crops n.e.c."),
    "Other legumes for forage": ("aff", "Growing crops n.e.c."),

    # ===== TOBACCO (-> oac) =====
    "Unmanufactured tobacco": ("oac", "Growing tobacco"),
    
    # ===== RUBBER (-> oac) =====
    "Natural rubber in primary forms": ("oac", "Growing crops n.e.c."),
    
    # ===== SPICES (-> oac or pec) =====
    "Anise, badian, coriander, cumin, caraway, fennel and juniper berries, raw": ("oac", "Growing spices, aromatic, drug and pharmaceutical crops"),
    "Chillies and peppers, dry (Capsicum spp., Pimenta spp.), raw": ("oac", "Growing spices, aromatic, drug and pharmaceutical crops"),
    "Other stimulant, spice and aromatic crops, n.e.c.": ("oac", "Growing spices, aromatic, drug and pharmaceutical crops"),
    "Pepper (Piper spp.), raw": ("pec", "Growing spices, aromatic, drug and pharmaceutical crops"),
    "Cloves (whole stems), raw": ("pec", "Growing spices, aromatic, drug and pharmaceutical crops"),
    "Nutmeg, mace, cardamoms, raw": ("pec", "Growing spices, aromatic, drug and pharmaceutical crops"),
    "Ginger, raw": ("oac", "Growing spices, aromatic, drug and pharmaceutical crops"),
    "Cinnamon and cinnamon-tree flowers, raw": ("pec", "Growing spices, aromatic, drug and pharmaceutical crops"),
    "Vanilla, raw": ("pec", "Growing spices, aromatic, drug and pharmaceutical crops"),
    "Peppermint, spearmint": ("oac", "Growing spices, aromatic, drug and pharmaceutical crops"),
    "Pyrethrum, dried flowers": ("oac", "Growing spices, aromatic, drug and pharmaceutical crops"),
    
    # ===== FODDER/FORAGE (-> oac) =====
    "Beets for fodder": ("oac", "Growing crops n.e.c."),
    "Swedes for fodder": ("oac", "Growing crops n.e.c."),
    "Vegetables and roots fodder": ("oac", "Growing crops n.e.c."),
    "Forage and silage, maize": ("oac", "Growing crops n.e.c."),
    "Other forage products, n.e.c.": ("oac", "Growing crops n.e.c."),
    "Other grasses for forage": ("oac", "Growing crops n.e.c."),
    "Turnips for forage": ("oac", "Growing crops n.e.c."),
    "Forage and silage, rye grass": ("oac", "Growing crops n.e.c."),
    "Forage and silage, sorghum": ("oac", "Growing crops n.e.c."),
    "Mixed Grasses and Legumes": ("oac", "Growing crops n.e.c."),
    "Forage and silage, green oilseeds": ("oac", "Growing crops n.e.c."),
    "Cabbage for fodder": ("oac", "Growing crops n.e.c."),
    "Carrots for fodder": ("oac", "Growing crops n.e.c."),

    "Hop cones": ("oac", "Growing crops n.e.c."),

    # ===== PERENNIAL EVERGREEN CROPS (pec) =====
    # Tea
    "Tea leaves": ("pec", "Growing beverage crops (coffee, tea etc)"),
    "Maté leaves": ("pec", "Growing beverage crops (coffee, tea etc)"),
    # Coconut
    "Coconuts, in shell": ("pec", "Growing fruits and nuts"),
    # Citrus
    "Oranges": ("pec", "Growing fruits and nuts"),
    "Tangerines, mandarins, clementines": ("pec", "Growing fruits and nuts"),
    "Lemons and limes": ("pec", "Growing fruits and nuts"),
    "Pomelos and grapefruits": ("pec", "Growing fruits and nuts"),
    "Other citrus fruit, n.e.c.": ("pec", "Growing fruits and nuts"),
    # Tropical fruits
    "Avocados": ("pec", "Growing fruits and nuts"),
    "Mangoes, guavas and mangosteens": ("pec", "Growing fruits and nuts"),
    "Bananas": ("pec", "Growing fruits and nuts"),
    "Plantains and cooking bananas": ("pec", "Growing fruits and nuts"),
    "Pineapples": ("pec", "Growing fruits and nuts"),
    "Papayas": ("pec", "Growing fruits and nuts"),
    "Other tropical fruits, n.e.c.": ("pec", "Growing fruits and nuts"),
    "Dates": ("pec", "Growing fruits and nuts"),
    # Tropical nuts
    "Cashew nuts, in shell": ("pec", "Growing fruits and nuts"),
    "Areca nuts": ("pec", "Growing fruits and nuts"),
    "Kola nuts": ("pec", "Growing fruits and nuts"),
    "Karite nuts (sheanuts)": ("pec", "Growing fruits and nuts"),
    "Cashewapple": ("pec", "Growing fruits and nuts"),
    
    # ===== PERENNIAL DECIDUOUS CROPS (pdc) =====
    # Grapes
    "Grapes": ("pdc", "Growing grapes"),
    # Olives
    "Olives": ("pdc", "Growing fruits and nuts"),
    # Pome fruits
    "Apples": ("pdc", "Growing fruits and nuts"),
    "Pears": ("pdc", "Growing fruits and nuts"),
    "Quinces": ("pdc", "Growing fruits and nuts"),
    "Other pome fruits": ("pdc", "Growing fruits and nuts"),
    # Stone fruits
    "Peaches and nectarines": ("pdc", "Growing fruits and nuts"),
    "Plums and sloes": ("pdc", "Growing fruits and nuts"),
    "Apricots": ("pdc", "Growing fruits and nuts"),
    "Cherries": ("pdc", "Growing fruits and nuts"),
    "Sour cherries": ("pdc", "Growing fruits and nuts"),
    "Other stone fruits": ("pdc", "Growing fruits and nuts"),
    # Other fruits
    "Figs": ("pdc", "Growing fruits and nuts"),
    "Persimmons": ("pdc", "Growing fruits and nuts"),
    "Kiwi fruit": ("pdc", "Growing fruits and nuts"),
    "Other fruits, n.e.c.": ("pdc", "Growing fruits and nuts"),
    "Locust beans (carobs)": ("pdc", "Growing fruits and nuts"),
    # Berries
    "Strawberries": ("pdc", "Growing fruits and nuts"),
    "Raspberries": ("pdc", "Growing fruits and nuts"),
    "Blueberries": ("pdc", "Growing fruits and nuts"),
    "Cranberries": ("pdc", "Growing fruits and nuts"),
    "Currants": ("pdc", "Growing fruits and nuts"),
    "Gooseberries": ("pdc", "Growing fruits and nuts"),
    "Other berries and fruits of the genus vaccinium n.e.c.": ("pdc", "Growing fruits and nuts"),
    # Nuts
    "Almonds, in shell": ("pdc", "Growing fruits and nuts"),
    "Walnuts, in shell": ("pdc", "Growing fruits and nuts"),
    "Pistachios, in shell": ("pdc", "Growing fruits and nuts"),
    "Hazelnuts, in shell": ("pdc", "Growing fruits and nuts"),
    "Chestnuts, in shell": ("pdc", "Growing fruits and nuts"),
    "Brazil nuts, in shell": ("pdc", "Growing fruits and nuts"),
    "Other nuts (excluding wild edible nuts and groundnuts), in shell, n.e.c.": ("pdc", "Growing fruits and nuts"),
}



def generate_sector_mapping(
    output_csv: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Write the 175 -> 27 -> 14 mapping to CSV and return the path."""
    logger = logger or logging.getLogger(__name__)
    df = pd.DataFrame(
        [{"crop_175": crop, "crop_27": mapping[0], "sector_14": mapping[1]}
         for crop, mapping in crop_mapping.items()]
    )
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    logger.info(f"  total crops mapped: {len(df)}")
    logger.info(f"  unique crop_27 categories: {df['crop_27'].nunique()}")
    logger.info(f"  unique sector_14 sectors: {df['sector_14'].nunique()}")
    logger.info(f"  saved: {out}")
    return str(out)
