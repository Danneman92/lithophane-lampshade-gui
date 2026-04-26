# Lithophane Lampshade GUI

A PyQt5 / OpenGL desktop application for generating 3-D printable lithophane lampshades from photos.

## Features

- **Three shade types** – Normal (cone/cylinder), Sphere, Flat panel
- **Multi-panel** – 1–12 panels, each with its own photo
- **Top & bottom brims** – configurable height and radial thickness
- **Frames / pillars** – vertical separators between panels, now sitting *flush* against the lithophane surface (no gap)
- **Sprocket teeth** – optional snap-together teeth on the top brim ring (tooth count, height, arc-width all configurable)
- **Lamp socket adapter** – optional hollow cylinder above the top brim that slides over the bulb fitting (bore diameter, wall thickness, height all configurable; default 26 mm bore suits E27)
- **Live 3-D preview** – drag to rotate, scroll to zoom, hi/lo-res swap during interaction
- **Binary STL export**

## Installation

```bash
pip install -r requirements.txt
python app.py
```

## Requirements

- Python 3.8+
- PyQt5 ≥ 5.15
- numpy ≥ 1.19
- Pillow ≥ 8.0
- PyOpenGL ≥ 3.1

## Frame gap fix

The previous version used a `clearance` offset so frames floated away from the panel surface.
The new version samples the actual outer-shell radius at each pillar height and places the
inner face directly against it — zero gap by default.

## Sprocket teeth

Enable in the **Sprocket Teeth** panel. Print two matching rings with `tooth_w_frac ≈ 0.45`
so teeth and gaps are nearly equal — they snap together to hold the lampshade segments in alignment.

## Lamp socket adapter

Enable in the **Lamp Socket Adapter** panel. Set **Inner Bore Diam** to match your bulb holder:
- E27 standard neck ≈ 26 mm
- E14 small screw ≈ 17 mm
- GU10 bayonet ≈ 25 mm

The adapter rises above the top brim and grips the fitting by friction fit.
Increase **Wall Thickness** (default 2.5 mm) for a stiffer grip.
