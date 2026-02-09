from __future__ import annotations

from pathlib import Path


def _load_deps():
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Pillow is required to generate demo assets") from exc
    return Image, ImageDraw


def _draw_rain(image, draw):  # type: ignore[no-untyped-def]
    w, h = image.size
    for x in range(0, w, 12):
        for y in range(0, h, 24):
            draw.line((x, y, x - 6, y + 16), fill=(170, 200, 255), width=2)
    draw.rectangle((100, 320, 410, 500), fill=(40, 40, 60))
    draw.ellipse((220, 180, 300, 260), fill=(240, 210, 190))
    draw.rectangle((205, 255, 315, 420), fill=(30, 50, 80))


def _draw_robot(image, draw):  # type: ignore[no-untyped-def]
    draw.rectangle((120, 120, 392, 392), fill=(80, 85, 95), outline=(200, 200, 210), width=4)
    draw.rectangle((180, 180, 330, 260), fill=(30, 40, 55))
    draw.ellipse((210, 200, 245, 235), fill=(120, 220, 255))
    draw.ellipse((275, 200, 310, 235), fill=(120, 220, 255))
    draw.rectangle((230, 285, 285, 315), fill=(180, 180, 180))
    draw.rectangle((30, 230, 120, 280), fill=(95, 95, 105))
    draw.rectangle((392, 230, 482, 280), fill=(95, 95, 105))


def _draw_city(image, draw):  # type: ignore[no-untyped-def]
    w, h = image.size
    draw.rectangle((0, h * 0.6, w, h), fill=(28, 34, 52))
    x = 24
    while x < w - 40:
        width = 26 + (x % 40)
        top = 120 + (x % 160)
        draw.rectangle((x, top, x + width, h * 0.6), fill=(45, 52, 75))
        y = top + 8
        while y < h * 0.6 - 12:
            draw.rectangle((x + 6, y, x + 10, y + 5), fill=(238, 220, 140))
            draw.rectangle((x + 15, y, x + 19, y + 5), fill=(238, 220, 140))
            y += 14
        x += width + 12


def _build_canvas(color_top: tuple[int, int, int], color_bottom: tuple[int, int, int]):  # type: ignore[no-untyped-def]
    Image, _ = _load_deps()
    image = Image.new("RGB", (512, 512), color_top)
    pixels = image.load()
    for y in range(512):
        t = y / 511.0
        r = int(color_top[0] * (1 - t) + color_bottom[0] * t)
        g = int(color_top[1] * (1 - t) + color_bottom[1] * t)
        b = int(color_top[2] * (1 - t) + color_bottom[2] * t)
        for x in range(512):
            pixels[x, y] = (r, g, b)
    return image


def main() -> int:
    Image, ImageDraw = _load_deps()
    root = Path("assets/init")
    root.mkdir(parents=True, exist_ok=True)

    rain = _build_canvas((35, 45, 78), (18, 22, 35))
    draw = ImageDraw.Draw(rain)
    _draw_rain(rain, draw)
    rain.save(root / "rain.png")

    robot = _build_canvas((28, 30, 36), (12, 14, 18))
    draw = ImageDraw.Draw(robot)
    _draw_robot(robot, draw)
    robot.save(root / "robot.png")

    city = _build_canvas((88, 122, 188), (225, 164, 98))
    draw = ImageDraw.Draw(city)
    _draw_city(city, draw)
    city.save(root / "city.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
