# Lithophane Lampshade GUI

A Python GUI application for creating custom lithophane lampshades from digital images.

## Overview

This application provides an intuitive graphical interface for converting your favorite photos into lithophane designs suitable for 3D printing as cylindrical lampshades. Lithophanes are thin, translucent objects that show detailed images when illuminated from behind.

## Features

- **Easy Image Selection**: Browse and select images from your computer
- **Customizable Dimensions**: Set lampshade diameter, height, and thickness
- **Real-time Preview**: Visual feedback during the design process
- **Progress Tracking**: Built-in progress bar for generation status
- **User-friendly Interface**: Clean, intuitive GUI built with Python tkinter

## Requirements

See `requirements.txt` for the full list of dependencies. Main requirements include:

- Python 3.7+
- Pillow (PIL) for image processing
- NumPy for numerical computations
- tkinter (included with Python)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Danneman92/lithophane-lampshade-gui.git
   cd lithophane-lampshade-gui
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. **Select Image**: Click "Browse" to choose your image file
   - Supported formats: PNG, JPG, JPEG, BMP, TIFF

3. **Configure Settings**:
   - **Diameter**: Set the lampshade diameter in millimeters (default: 150mm)
   - **Height**: Set the lampshade height in millimeters (default: 200mm)
   - **Thickness**: Set the wall thickness in millimeters (default: 2.5mm)

4. **Generate**: Click "Generate Lithophane" to create your design

## Technical Details

- **Image Processing**: Converts color images to grayscale for lithophane effect
- **Cylindrical Mapping**: Wraps flat images around cylindrical surfaces
- **Thickness Variation**: Darker areas become thicker, lighter areas thinner
- **3D Print Ready**: Outputs optimized for FDM 3D printing

## Supported Image Formats

- PNG
- JPEG/JPG
- BMP
- TIFF

## Tips for Best Results

1. **High Contrast Images**: Work best for lithophanes
2. **Resolution**: Higher resolution images produce finer detail
3. **Aspect Ratio**: Consider your lampshade dimensions when selecting images
4. **Content**: Portraits and landscapes with good lighting work well

## Development

This project is structured as follows:

```
lithophane-lampshade-gui/
├── main.py           # Main application entry point
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

## License

This project is open source. Please check the repository for license details.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

*Created for makers, designers, and anyone who loves personalized lighting!*
