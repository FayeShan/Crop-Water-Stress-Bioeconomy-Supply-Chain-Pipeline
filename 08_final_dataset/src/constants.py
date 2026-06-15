"""Crop codes, display names and GeoTIFF band order for the published dataset."""

from rasterio.transform import Affine

# 27 crop categories (NetCDF variable names)
CROPS = [
    "aff", "bar", "bea", "cas", "ckp", "coc", "cof", "cot", "cwp",
    "mai", "mil", "nut", "oac", "pdc", "pec", "plm", "pot", "rap",
    "ri1", "ri2", "sgb", "sgc", "sor", "soy", "sun", "vgt", "wh",
]

CROP_NAMES = {
    "aff": "Alfalfa", "bar": "Barley", "bea": "Beans",
    "cas": "Cassava", "ckp": "Chick peas", "coc": "Cocoa beans",
    "cof": "Coffee", "cot": "Seed cotton", "cwp": "Cow peas",
    "mai": "Maize (corn)", "mil": "Millet", "nut": "Groundnuts",
    "oac": "Other annual crops", "pdc": "Perennial deciduous crops",
    "pec": "Perennial evergreen crops", "plm": "Oil palm fruit",
    "pot": "Potatoes", "rap": "Rape or colza seed",
    "ri1": "Rice (main season)", "ri2": "Rice (second season)",
    "sgb": "Sugar beet", "sgc": "Sugar cane", "sor": "Sorghum",
    "soy": "Soya beans", "sun": "Sunflower seed", "vgt": "Vegetables",
    "wh": "Wheat",
}

# Band order used in the published GeoTIFFs (kept fixed for backward compatibility)
GEOTIFF_BAND_ORDER = [
    "aff", "bar", "bea", "ckp", "cot", "cwp", "mai", "sgb", "sgc",
    "sor", "soy", "cof", "nut", "rap", "vgt", "pot", "sun", "wh",
    "ri1", "ri2", "mil", "cas", "plm", "coc", "pec", "pdc", "oac",
]

# 5-arcmin global grid affine transform (top-left origin, 0.0833 deg)
GEOTIFF_TRANSFORM = Affine(0.0833330000000103, 0.0, -180.0,
                           0.0, -0.08333299999999612, 90.0)
