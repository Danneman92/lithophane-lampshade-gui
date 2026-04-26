# Lithophane Lampshade GUI

A PyQt5 / OpenGL desktop application for generating 3-D printable lithophane lampshades from photos.

## Features

- **Three shade types** – Normal (cone/cylinder), Sphere, Flat panel
- **Multi-panel** – 1–12 panels, each with its own photo
- **Top & bottom brims** – configurable height and radial thickness
- **Frames / pillars** – vertical separators between panels, sitting *flush* against the lithophane surface (no gap)
- **Lamp socket adapter** – hollow cylinder that sits *inside* the shade with its bottom flush with the shade bottom. Slides over the bulb fitting. Includes a configurable outward lip (stop collar) so it cannot be pushed up through the fitting
- **Spokes** – optional radial ribs bridging the socket tube to the inner shade wall
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

## Lamp socket adapter

Enable in the **Lamp Socket Adapter** group.

| Parameter | Description |
|---|---|
| Inner Bore Diam | Match your bulb holder: E27 ≈ 26 mm, E14 ≈ 17 mm, GU10 ≈ 25 mm |
| Wall Thickness | Default 2.5 mm |
| Adapter Height | How far the tube reaches up inside the shade (default 60 mm) |
| Lip Height | Height of the stop-collar at the bottom (default 4 mm) |
| Lip Overhang | How far the collar flares outward beyond the tube wall (default 4 mm) |

The socket bottom is at **y = 0** (same plane as the shade bottom), so the lip rests on the bulb fitting housing and the tube points upward into the shade.

## Spokes

Enable in the same **Lamp Socket Adapter** group.
Spokes are flat rectangular ribs at y = 0 that radiate from the socket tube outward
to the inner wall of the shade.

| Parameter | Description |
|---|---|
| Spoke Count | Number of equally-spaced spokes (default 4) |
| Spoke Width | Tangential width of each spoke in mm (default 4 mm) |
| Spoke Thickness | Vertical thickness of each spoke in mm (default 2 mm) |

## Frame gap fix

The previous version used a clearance offset so frames floated away from the panel surface.
The new version samples the actual outer-shell radius and places the inner face directly
against it — zero gap by default.
