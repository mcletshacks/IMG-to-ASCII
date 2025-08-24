from __future__ import annotations

import concurrent.futures
import datetime as _dt
import functools
import sys
import os
import re

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

try:
    from PIL import Image, ImageOps

except Exception:
    print("Error (1) ~~ Make sure u installed the requirements!")
    sys.exit(1)

class AnotherOne(type):
    instances: Dict[type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.instances:
            cls.instances[cls] = super().__call__(*args, **kwargs)

        return cls.instances[cls]

def curr_time() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def dir_check(p: Path) -> Path:
    p.mkdir(parents = True, exist_ok = True)

    return p

def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

class CharsetKind(Enum):
    Easy = "simple"
    Hard = "complex"

class CharRamp(metaclass = AnotherOne):
    def __init__(self, chars: str):
        if len(chars) < 2:
            raise ValueError("Error (2) ~~ In Func CharRamp -> requires atleast 2 characters")

        self.chars_value = chars

    @property
    def chars(self) -> str:
        return self.chars_value

    def maybe_invert(self, invert: bool) -> "CharRamp":
        if not invert:
            return self.__class__(self.chars_value)

        return self.__class__(self.chars_value[::-1])

class EasyRamp(CharRamp):
    def __init__(self, chars: str = "@%#*+=-:. "):
        super().__init__(chars)

class HardRamp(CharRamp):
    def __init__(self, chars: str = "$@B%8&WM#*oahkbdpqwmZ0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\\\"^`'. "):
        super().__init__(chars)

def get_ramp(kind: CharsetKind) -> CharRamp:
    if kind is CharsetKind.Easy:
        return EasyRamp()

    if kind is CharsetKind.Hard:
        return HardRamp()

    raise ValueError("Error (3) ~~ unsupported charset")

@dataclass(frozen = True)
class PipelineConfig:
    aspect_correction: float = 0.55
    color: bool = False

class ImageLoader(metaclass = AnotherOne):
    def open(self, path: Path) -> "Image.Image":
        try:
            img = Image.open(path)

        except Exception as e:
            raise RuntimeError(f"Error (4) ~~ Couldnt open image: {e}")

        try:
            img = ImageOps.exif_transpose(img)

        except Exception:
            pass

        return img

class SizeMath:
    def __init__(self, aspect_correction: float):
        self.aspect_correction = aspect_correction

    def doMath(self, img_w: int, img_h: int, width: int) -> Tuple[int, int]:
        ac = self.aspect_correction

        h = int(img_h * (width / img_w) * ac)

        return max(1, width), max(1, h)

@functools.lru_cache(maxsize = 1024)
def map_pixel_to_char(val: int, ramp_chars: str) -> str:
    idx = int(val * (len(ramp_chars) - 1) / 255)

    return ramp_chars[idx]

class MainRender:
    def __init__(self, ramp: CharRamp, config: PipelineConfig):
        self.ramp = ramp
        self.config = config

    def render(self, img: "Image.Image", target_w: int, target_h: int) -> str:
        img_gray = img.convert("L").resize((target_w, target_h), Image.BICUBIC)
        img_color = img.convert("RGB").resize((target_w, target_h), Image.BICUBIC) if self.config.color else None

        pixels = img_gray.getdata()
        pixels_wcolor = img_color.getdata() if img_color else None

        lines: List[str] = []

        for y in range(target_h):
            row_chars: List[str] = []
            base_idx = y * target_w

            for x in range(target_w):
                idx = base_idx + x
                val = pixels[idx]
                ch = map_pixel_to_char(val, self.ramp.chars)

                if pixels_wcolor is not None:
                    r, g, b = pixels_wcolor[idx]
                    row_chars.append(f"\x1b[38;2;{r};{g};{b}m{ch}\x1b[0m")

                else:
                    row_chars.append(ch)

            lines.append("".join(row_chars))

        return "\n".join(lines)

class BombasticBuilder:
    def __init__(self):
        self.loader = ImageLoader()
        self.size_comp = SizeMath(0.55)

        self.widths = [24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 200, 240, 280, 320]
        self.charsets = [CharsetKind.Easy, CharsetKind.Hard]
        self.inversions = [False, True]
        self.colors = [False, True]

    def target_dir(self, root: Path, stem: str, color: bool, charset: CharsetKind, invert: bool, width: int) -> Path:
        a = "color" if color else "grayscale"
        b = "hard" if charset is CharsetKind.Hard else "easy"
        c = "inverted" if invert else "normal"
        d = f"w{width:03d}"

        return root / stem / a / b / c / d

    def first_process(self, img_path: Path, out_root: Path) -> List[Path]:
        img = self.loader.open(img_path)
        stem = sanitize_name(img_path.stem)

        results: List[Path] = []
        tasks: List[Tuple[bool, CharsetKind, bool, int]] = []

        for color in self.colors:
            for cs in self.charsets:
                for inv in self.inversions:
                    for w in self.widths:
                        tasks.append((color, cs, inv, w))

        def run_task(t: Tuple[bool, CharsetKind, bool, int]) -> Path:
            color, cs, inv, w = t

            ramp = get_ramp(cs).maybe_invert(inv)
            cfg = PipelineConfig(color = color)

            tw, th = self.size_comp.doMath(img.width, img.height, w)
            art = MainRender(ramp, cfg).render(img, tw, th)

            td = self.target_dir(out_root, stem, color, cs, inv, w)
            dir_check(td)

            fp = td / "art.txt"
            with open(fp, "w", encoding = "utf-8", newline = "\n") as f:
                f.write(art)

            return fp

        with concurrent.futures.ThreadPoolExecutor(max_workers = min(32, os.cpu_count() or 4)) as ex:
            for p in ex.map(run_task, tasks):
                results.append(p)

        return results

def find_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}

    if not root.exists():
        dir_check(root)

        return []

    r: List[Path] = []

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            r.append(p)

    return r

def main() -> int:
    in_root = Path("input")
    out_root = Path("output")

    imgs = find_images(in_root)

    if not imgs:
        print("Error (5) ~~ Couldnt find any images, put them inside input folder and rerun")

        return 0

    bb = BombasticBuilder()
    all_paths: List[Path] = []

    for img in imgs:
        try:
            paths = bb.first_process(img, out_root)
            all_paths.extend(paths)

        except Exception as e:
            print(f"Error (5.3) ~~ Couldnt process {img}: {e}", file = sys.stderr)

    print("==============================================")
    print(f"Input Files: {len(imgs)}")
    print(f"Files created: {len(all_paths)}")
    print(f"Output in: {out_root.resolve()}")
    print("==============================================")

    for p in sorted(all_paths):
        print(f"- {p}")

    return 0

if __name__ == "__main__":
    sys.exit(main())