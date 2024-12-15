# Camera Image Signal Processing Pipeline

## Overview
This project implements a comprehensive **Camera Image Signal Processing (ISP) Pipeline** to process RAW images and generate final RGB images. The pipeline performs multiple steps such as black level subtraction, lens shading correction, demosaicing, white balance, color correction, tone mapping, and gamma correction. Additionally, it provides functionalities for saving intermediate and final outputs.

## Features
- **Metadata Extraction**: Extracts metadata from RAW files, including CFA patterns and thumbnails.
- **Black Level Subtraction**: Removes the black level and normalizes pixel values.
- **Lens Shading Correction**: Applies correction for lens shading using an optional map.
- **Demosaicing**: Converts Bayer domain data into an RGB image.
- **White Balance & Color Correction**: Adjusts colors using white balance multipliers and a color correction matrix (CCM).
- **Tone Mapping**: Maps tones to preserve dynamic range.
- **Gamma Correction**: Applies sRGB gamma correction to the image.
- **Image Saving**: Saves intermediate and final results in various formats.

## Requirements
### Python Libraries
- `rawpy`
- `imageio`
- `numpy`
- `argparse`
- `Pillow`
- `colour-demosaicing`
- `scipy`

Install the required libraries using:
```bash
pip install rawpy imageio numpy pillow colour-demosaicing scipy
```

## Usage
### Command-line Arguments
The pipeline can be run with various options:

| Argument        | Type    | Description                                               |
|-----------------|---------|-----------------------------------------------------------|
| `-sg`          | Flag    | Save Bayer-domain image as grayscale image.               |
| `-sb`          | Flag    | Save Bayer-domain image as RGB image.                    |
| `-sd`          | Flag    | Save demosaiced image.                                    |
| `-sc`          | Flag    | Save final RGB image.                                     |
| `-et`          | Flag    | Extract thumbnail JPEG if available.                     |
| `-im`          | String  | Path to the input RAW image (required).                  |
| `-lm`          | String  | Path to an optional lens shading correction map.          |
| `-cm`          | String  | Path to an optional color correction matrix (CCM).        |

### Example Usage
To run the pipeline with an input image and save the final RGB output:
```bash
python isp.py -im input_image.raw -sc
```
To save intermediate grayscale and Bayer RGB images:
```bash
python isp.py -im input_image.raw -sg -sb
```

### Output Files
The pipeline generates the following outputs based on the provided arguments:
- `grayscale.png`: Normalized Bayer-domain grayscale image.
- `bayer_rgb.png`: Bayer-domain image converted to RGB.
- `demosaiced.png`: RGB image after demosaicing.
- `final_rgb.png`: Fully processed RGB image.

## Pipeline Workflow
1. **Load RAW Image**: Load the RAW image using `rawpy`.
2. **Extract Metadata**: Extract CFA pattern, color indices, and optional thumbnail.
3. **Black Level Subtraction**: Normalize pixel values by subtracting the black level.
4. **Lens Shading Correction** (optional): Correct pixel values using a lens shading map.
5. **Demosaicing**: Convert Bayer CFA data to an RGB image using bilinear interpolation.
6. **White Balance and Color Correction**: Apply camera-specific white balance and CCM.
7. **Tone Mapping**: Adjust tone for better visual appearance and dynamic range compression.
8. **Gamma Correction**: Apply gamma correction to produce the final sRGB image.
9. **Save Images**: Save the intermediate and final images based on user inputs.

## File Structure
- `isp.py`: Main script for the pipeline.
- `flower.dng`, `person.dng`, `tree.dng`: Example input RAW image (not included).
- `lens_shading_map.png`: Example optional lens shading correction map.
- `color_correction_matrix.txt`: Example optional color correction matrix.

## Future Enhancements
- Add support for advanced tone mapping operators.
- Include noise reduction and sharpening steps.
- Extend compatibility to additional RAW file formats.

## References
- [rawpy Documentation](https://letmaik.github.io/rawpy/)
- [colour-demosaicing](https://colour-demosaicing.readthedocs.io/en/latest/)
- [Pillow Documentation](https://pillow.readthedocs.io/)
- [sRGB Gamma Correction](https://en.wikipedia.org/wiki/SRGB)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

