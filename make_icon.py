"""Generate sigmoid_explorer.ico — a square app icon showing the brand sigmoid.

Renders a white asymmetric-sigmoid curve on the brand-blue rounded square at
high resolution, then writes a multi-resolution Windows .ico. Run with:
    uv run python make_icon.py
"""

import numpy as np
from PIL import Image, ImageDraw

BLUE = (0, 102, 204)        # #0066cc — same blue as the splash curve
WHITE = (255, 255, 255)
SS = 4                      # supersample factor for crisp downscaling
SIZE = 256 * SS             # render canvas, downscaled to ICO sizes below


def sigmoid(x, k=11.0, x0=0.5, nu=0.45):
    """Asymmetric sigmoid in [0,1] -> [0,1], matching the app's curve shape."""
    z = np.clip(-k * (x - x0), -500.0, 500.0)
    return np.exp(-np.log1p(np.exp(z)) / nu)


def main():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    radius = int(SIZE * 0.20)
    d.rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=radius, fill=BLUE)

    # Curve laid out with margins; y flipped so it rises left-to-right.
    mx, my = 0.16 * SIZE, 0.26 * SIZE
    xs = np.linspace(0.0, 1.0, 400)
    ys = sigmoid(xs)
    px = mx + xs * (SIZE - 2 * mx)
    py = (SIZE - my) - ys * (SIZE - 2 * my)
    pts = list(zip(px.tolist(), py.tolist()))

    # Stamp circles along a densely-sampled path: yields a smooth, uniformly
    # rounded stroke with no segment-overlap artifacts (and free round caps).
    w = int(SIZE * 0.075)
    dense = np.linspace(0.0, 1.0, 2000)
    dpx = mx + dense * (SIZE - 2 * mx)
    dpy = (SIZE - my) - sigmoid(dense) * (SIZE - 2 * my)
    r = w / 2
    for cx, cy in zip(dpx.tolist(), dpy.tolist()):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=WHITE)

    base = img.resize((256, 256), Image.LANCZOS)
    base.save("sigmoid_explorer.ico",
              sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                     (128, 128), (256, 256)])
    print("Wrote sigmoid_explorer.ico")


if __name__ == "__main__":
    main()
