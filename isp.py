import rawpy
import imageio
import numpy as np
import sys
import argparse
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from scipy import interpolate

# Set print options for debugging purposes
np.set_printoptions(threshold=sys.maxsize)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Image Signal Processing Pipeline")
    parser.add_argument('-sg', action='store_true', help='Save Bayer-domain image as grayscale image')
    parser.add_argument('-sb', action='store_true', help='Save Bayer-domain image as RGB image')
    parser.add_argument('-sd', action='store_true', help='Save demosaiced image')
    parser.add_argument('-sc', action='store_true', help='Save final RGB image')
    parser.add_argument('-et', action='store_true', help='Extract thumbnail JPEG if exists')
    parser.add_argument('-im', type=str, required=True, help='Input image path')
    parser.add_argument('-lm', type=str, required=False, help='Optional lens shading correction map path')
    parser.add_argument('-cm', type=str, required=False, help='Optional color correction matrix path')
    return parser.parse_args()

class ImageSignalProcessing:
    def __init__(self, args: argparse.Namespace):
        print("Initializing Image Signal Processing...")
        self.args = args
        self.raw = rawpy.imread(self.args.im)
        self.raw_img = self.raw.raw_image.astype(np.float32)
        print(f"Raw image shape: {self.raw_img.shape}")

        # Load optional lens shading map
        if self.args.lm:
            self.lens_sm = imageio.imread(self.args.lm)
            print(f"Lens shading map shape: {self.lens_sm.shape}")
        else:
            self.lens_sm = None

    def create_cfa_indices(self):
        """Generate CFA indices for Bayer pattern."""
        print("Creating CFA indices...")
        cfa_pattern_id = np.array(self.raw.raw_pattern)
        color_desc = np.frombuffer(self.raw.color_desc, dtype=np.byte)
        tile_pattern = np.array([[chr(color_desc[cfa_pattern_id[0, 0]]), chr(color_desc[cfa_pattern_id[0, 1]])],
                                 [chr(color_desc[cfa_pattern_id[1, 0]]), chr(color_desc[cfa_pattern_id[1, 1]])]], dtype=object)
        self.cfa_pattern_rgb = tile_pattern.copy()

        # Expand G to GR and GB for lens shading correction
        for i in range(2):
            for j in range(2):
                if tile_pattern[i, j] == 'G':
                    tile_pattern[i, j] = 'G' + tile_pattern[i, (j + 1) % 2]
        print(tile_pattern)
        self.raw_color_index = np.tile(tile_pattern, (self.raw_img.shape[0] // 2, self.raw_img.shape[1] // 2))

    def extract_metadata(self):
        """Extract metadata from the RAW image."""
        print("Extracting metadata...")
        self.create_cfa_indices()
        if self.args.et:
            self.extract_thumbnail()

    def extract_thumbnail(self):
        """Extract and save thumbnail if available."""
        print("Extracting thumbnail...")
        try:
            thumb = self.raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                with open('thumb.jpg', 'wb') as f:
                    f.write(thumb.data)
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                imageio.imsave('thumb.tiff', thumb.data)
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            print("No thumbnail found or unsupported format.")

    def subtract_black_level(self):
        """Subtract black level and normalize the RAW image."""
        print("Subtracting black level...")
        black = np.tile(np.reshape(self.raw.black_level_per_channel, (2, 2)), (self.raw_img.shape[0] // 2, self.raw_img.shape[1] // 2))
        self.gs_img = (self.raw_img - black) / (self.raw.white_level - black)

    # create bayer-domain raw image that can be displayed as RGB image
    def gen_bayer_rgb_img(self):
        ##https://stackoverflow.com/questions/19766757/replacing-numpy-elements-if-condition-is-met
        r_channel = np.where(self.raw_color_index == 'R', self.gs_img, 0)
        g_channel = np.where(((self.raw_color_index == 'GR') | (self.raw_color_index == 'GB')), self.gs_img, 0)
        b_channel = np.where(self.raw_color_index == 'B', self.gs_img, 0)

        #https://hausetutorials.netlify.app/posts/2019-12-20-numpy-reshape/
        self.bayer_color_img = np.stack((r_channel, g_channel, b_channel), axis=2)

    def lens_shading_correction(self):
        """Apply lens shading correction if a map is provided."""
        if self.lens_sm is None:
            return

        print("Applying lens shading correction...")
        x = np.linspace(0, self.raw_img.shape[0] - 1, self.lens_sm.shape[0])
        y = np.linspace(0, self.raw_img.shape[1] - 1, self.lens_sm.shape[1])

        corrected_maps = [
            interpolate.interp2d(y, x, self.lens_sm[:, :, i], kind='cubic')(np.arange(self.raw_img.shape[1]),
                                                                             np.arange(self.raw_img.shape[0]))
            for i in range(4)
        ]

        for color, correction in zip(['R', 'GR', 'GB', 'B'], corrected_maps):
            self.gs_img = np.where(self.raw_color_index == color, self.gs_img * correction, self.gs_img)

    def demosaic(self):
        """Perform demosaicing on the normalized image."""
        print("Demosaicing...")
        cfa_pattern = "".join(self.cfa_pattern_rgb.flatten())
        self.demosaic_img = demosaicing_CFA_Bayer_bilinear(self.gs_img, cfa_pattern)

    def apply_color_correction(self):
        """Apply white balance and color correction."""
        print("Applying white balance and color correction...")
        wb_matrix = np.diag(self.raw.camera_whitebalance[:3])
        cc_matrix = self.extract_ccm() if self.args.cm else self.raw.color_matrix[:3, :3]

        flat_img = self.demosaic_img.reshape(-1, 3).T
        corrected_img = np.clip(cc_matrix @ wb_matrix @ flat_img, 0, 1)
        self.color_img = corrected_img.T.reshape(*self.demosaic_img.shape)

    def extract_ccm(self) -> np.ndarray:
        """Extract color correction matrix from file."""
        with open(self.args.cm, 'r') as file:
            return np.array([float(x) for x in file.read().split()]).reshape(3, 3)
    
    def tone_mapping(self):
        """Apply tone mapping to the image."""
        print("Applying tone mapping...")
        luminance = 0.2126 * self.color_img[..., 0] + 0.7152 * self.color_img[..., 1] + 0.0722 * self.color_img[..., 2]
        key = np.mean(luminance)
        scale = 0.18 / (key + 1e-6)
        tone_mapped = self.color_img * scale
        self.color_img = np.clip(tone_mapped / (1 + tone_mapped), 0, 1)

    def apply_gamma_correction(self):
        """Apply gamma correction to the image."""
        print("Applying gamma correction...")
        mask = self.color_img < 0.0031308
        self.color_img[mask] *= 323 / 25
        self.color_img[~mask] = 211 / 200 * self.color_img[~mask] ** (5 / 12) - 11 / 200

    def save_images(self):
        """Save processed images based on command-line arguments."""
        print("Saving images...")
        if self.args.sg:
            Image.fromarray((self.gs_img * 255).astype(np.uint8)).save("grayscale.png")
        if self.args.sb:
            self.gen_bayer_rgb_img()
            Image.fromarray((self.bayer_color_img * 255).astype(np.uint8)).save("bayer_rgb.png")
        if self.args.sd:
            Image.fromarray((self.demosaic_img * 255).astype(np.uint8)).save("demosaiced.png")
        if self.args.sc:
            Image.fromarray((self.color_img * 255).astype(np.uint8)).save("final_rgb.png")


def main():
    args = parse_args()
    isp = ImageSignalProcessing(args)
    isp.extract_metadata()
    isp.subtract_black_level()
    isp.lens_shading_correction()
    isp.demosaic()
    isp.apply_color_correction()
    isp.tone_mapping()
    isp.apply_gamma_correction()
    isp.save_images()

if __name__ == "__main__":
    main()
