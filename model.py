import imageio
import numpy as np
import logging
import sys
from simple_lama_inpainting import SimpleLama
from PIL import Image

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.info("=== LaMa Backend Model Starting ===")

# Load model
simple_lama = SimpleLama()
logger.info("✅ SimpleLama model loaded successfully!") 

# Core processing
def call_lama(image: Image, mask: Image):
    image = image.convert("RGB")
    mask = mask.convert("L")
    result = simple_lama(image, mask)
    return np.array(result.convert("RGBA"))

def process_image(image: np.ndarray, mask_layer: np.ndarray | None):
    if mask_layer is None:
        return image
    mask_alpha = mask_layer[:, :, 3]
    binary_mask = (mask_alpha > 0).astype(np.uint8) * 255
    image_pil = Image.fromarray(image)
    mask_pil = Image.fromarray(binary_mask)
    return call_lama(image_pil, mask_pil)
