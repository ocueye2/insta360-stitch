#!/usr/bin/env python3
"""
insta360-stitch.py — Insta360 X3 dual-fisheye (.INSP / .DNG / .JPG) to
equirectangular panorama, in pure Python (numpy + Pillow).

What it does
------------
An X3 photo is NOT a panorama yet. It is two circular fisheye images
side by side in one file (5952x2976 for 18 MP, 11968x5984 for 72 MP):
the left circle is one lens, the right circle is the lens on the
opposite side of the camera, each covering ~200 degrees. This script
re-projects both circles onto a sphere and unwraps it as a 2:1
equirectangular image, feather-blending the overlap seam. The output
carries GPano XMP, so Google Photos / Facebook / 360 viewers recognise
it as a panorama.

    python3 stitch.py photo.insp
    python3 stitch.py ~/photos/x3/ --output-dir ~/photos/panos
    python3 stitch.py a.insp b.dng --width 5760 --jobs 4

Requirements: numpy, Pillow.  DNG additionally needs rawpy
(`pip install rawpy`). No OpenCV, no Insta360 Studio needed.

Calibration (measured on the sample IMG_20261109_174210_00_005.insp)
--------------------------------------------------------------------
* Layout: side-by-side, each circle diameter = image height,
  centres (H/2, H/2) and (3H/2, H/2), radius H/2.
* Projection: equidistant fisheye, r = R * theta / (FOV/2).
* FOV: Insta360 quotes 195 deg per lens. On the sample, 200 deg gave
  the cleanest seam (checked 190/195/200/205/210 — at 195 the door
  plaque left of the door ghosts, at 205 it ghosts the other way,
  at 200 it is single). Default is therefore 200; if your copy
  ghosts at the seam, try --fov 195 or --fov 205.
* Left circle = front lens, right circle = back lens. Swapping them
  is just a 180 deg yaw shift, so use --yaw 180 if your panorama
  comes out facing backwards.

Limits — read before blaming the code
-------------------------------------
* This is a geometric stitch, not Insta360 Studio's optical-flow
  stitch. Subjects closer than ~0.6 m (Insta360's safe stitching
  distance) sit at different depths in the two lenses (parallax),
  so they can ghost / double at the seam. Walls a metre or two away
  stitch cleanly; the selfie stick / tripod at the nadir is in the
  blind spot and may show as a smear at the very bottom.
* The two lenses can expose slightly differently; no gain matching
  is applied because on the sample the overlap-band gain ratio was
  ~1.00 — the brightness difference between the circles is the scene,
  not the lenses.
* DNG and 72 MP paths are implemented from the file layout, but only
  the 18 MP .INSP sample was verified end-to-end here. If a .DNG
  fails, install rawpy and re-run with the error it prints.
"""

import argparse
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_FOV = 200.0
SUPPORTED = {".insp", ".jpg", ".jpeg", ".dng", ".tif", ".tiff"}


# ---------------------------------------------------------------- loads

def load_image(path):
    """Return (rgb uint8 array HxWx3, exif bytes or None)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".dng":
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError(
                "DNG needs rawpy: pip install rawpy (then re-run)"
            ) from exc
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=8,
                no_auto_bright=False,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            )
        return np.asarray(rgb, dtype=np.uint8), None
    with Image.open(path) as im:
        exif = im.info.get("exif")
        return np.asarray(im.convert("RGB"), dtype=np.uint8), exif


def looks_like_dual_fisheye(rgb):
    """True if the four corners are (near-)black, as a dual-fisheye
    file's are. An already-stitched equirectangular file is not."""
    h, w = rgb.shape[:2]
    s = max(8, int(min(h, w) * 0.04))
    corners = [
        rgb[:s, :s], rgb[:s, -s:], rgb[-s:, :s], rgb[-s:, -s:],
    ]
    return all(c.mean() < 15 for c in corners)


def circle_geometry(rgb, layout):
    """Return (cx1, cy1, cx2, cy2, radius) for the two fisheye circles."""
    h, w = rgb.shape[:2]
    if layout in ("auto", "side-by-side"):
        if abs(w - 2 * h) > 2:
            raise ValueError(
                f"expected side-by-side dual fisheye (W == 2*H), got {w}x{h}"
            )
        return h / 2, h / 2, 3 * h / 2, h / 2, h / 2
    if layout == "top-bottom":
        if abs(h - 2 * w) > 2:
            raise ValueError(
                f"expected top-bottom dual fisheye (H == 2*W), got {w}x{h}"
            )
        return w / 2, w / 2, w / 2, 3 * w / 2, w / 2
    raise ValueError(f"unknown layout {layout!r}")


# ------------------------------------------------------------- sampling

def _bilinear(src, sx, sy):
    """Sample src (HxWx3 uint8) at float coords sx, sy. Returns float array."""
    h, w = src.shape[:2]
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    fx = (sx - x0).astype(np.float32)
    fy = (sy - y0).astype(np.float32)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    top = src[y0, x0] * (1 - fx[..., None]) + src[y0, x1] * fx[..., None]
    bot = src[y1, x0] * (1 - fx[..., None]) + src[y1, x1] * fx[..., None]
    return top * (1 - fy[..., None]) + bot * fy[..., None]


def stitch_array(rgb, out_w, fov_deg, yaw_deg, layout="auto",
                 radius_scale=1.0, chunk_rows=192):
    """Dual-fisheye RGB array -> equirectangular RGB array (out_w x out_w//2)."""
    src_h, src_w = rgb.shape[:2]
    cx1, cy1, cx2, cy2, radius = circle_geometry(rgb, layout)
    radius *= radius_scale
    out_h = out_w // 2
    half = math.radians(fov_deg) / 2.0
    yaw = math.radians(yaw_deg)

    out = np.empty((out_h, out_w, 3), dtype=np.uint8)
    # Longitude / latitude grids are built per chunk to bound memory.
    lon_all = (np.arange(out_w, dtype=np.float64) / out_w - 0.5) * 2 * math.pi + yaw
    sin_lon, cos_lon = np.sin(lon_all), np.cos(lon_all)

    for y0 in range(0, out_h, chunk_rows):
        y1 = min(y0 + chunk_rows, out_h)
        rows = y1 - y0
        lat = (0.5 - (np.arange(y0, y1, dtype=np.float64) + 0.5) / out_h) * math.pi
        sin_lat = np.sin(lat)[:, None]
        cos_lat = np.cos(lat)[:, None]
        # Direction vector for every output pixel in this chunk.
        dx = cos_lat * sin_lon[None, :]
        dy = np.repeat(sin_lat, out_w, axis=1)
        dz = cos_lat * cos_lon[None, :]
        del cos_lat

        theta_f = np.arccos(np.clip(dz, -1, 1))       # angle from front axis
        theta_b = math.pi - theta_f                   # angle from back axis
        phi_f = np.arctan2(dy, dx)                    # front lens: x = +dx
        phi_b = np.arctan2(dy, -dx)                   # back lens:  x = -dx (180 deg round Y)
        r_f = radius * theta_f / half
        r_b = radius * theta_b / half
        fx = cx1 + r_f * np.cos(phi_f)
        fy = cy1 - r_f * np.sin(phi_f)
        bx = cx2 + r_b * np.cos(phi_b)
        by = cy2 - r_b * np.sin(phi_b)
        del dx, dy, dz, phi_f, phi_b, r_f, r_b

        # Feather weight: 0 at the lens edge, max at its centre, so the
        # seam is a smooth cross-fade across the whole overlap band.
        wf = np.clip(half - theta_f, 0, None).astype(np.float32)
        wb = np.clip(half - theta_b, 0, None).astype(np.float32)
        tot = wf + wb
        valid = tot > 1e-6
        w = np.zeros_like(wf)
        w[valid] = wf[valid] / tot[valid]
        del wf, wb, tot, theta_f, theta_b

        a = _bilinear(rgb, fx, fy)
        b = _bilinear(rgb, bx, by)
        del fx, fy, bx, by
        blended = a * w[..., None] + b * (1.0 - w[..., None])
        del a, b, w
        blended[~valid] = 0
        out[y0:y1] = blended.astype(np.uint8)
    return out


# ------------------------------------------------------------------ XMP

GPANO_XMP = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="insta360-stitch.py">
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:GPano="http://ns.google.com/photos/1.0/panorama/">
<GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>
<GPano:CaptureSoftware>Insta360 X3</GPano:CaptureSoftware>
<GPano:StitchingSoftware>insta360-stitch.py</GPano:StitchingSoftware>
<GPano:ProjectionType>equirectangular</GPano:ProjectionType>
<GPano:PoseHeadingDegrees>0</GPano:PoseHeadingDegrees>
<GPano:InitialViewHeadingDegrees>0</GPano:InitialViewHeadingDegrees>
<GPano:InitialViewPitchDegrees>0</GPano:InitialViewPitchDegrees>
<GPano:InitialViewRollDegrees>0</GPano:InitialViewRollDegrees>
<GPano:FullPanoWidthPixels>{w}</GPano:FullPanoWidthPixels>
<GPano:FullPanoHeightPixels>{h}</GPano:FullPanoHeightPixels>
<GPano:CroppedAreaImageWidthPixels>{w}</GPano:CroppedAreaImageWidthPixels>
<GPano:CroppedAreaImageHeightPixels>{h}</GPano:CroppedAreaImageHeightPixels>
<GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels>
<GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels>
</rdf:Description>
</rdf:RDF>
</x:xmpmeta>"""


def inject_gpano_xmp(jpeg_path, width, height):
    """Insert a GPano XMP APP1 segment right after the JPEG SOI marker."""
    data = Path(jpeg_path).read_bytes()
    if not data.startswith(b"\xff\xd8"):
        return False
    payload = b"http://ns.adobe.com/xap/1.0/\x00" + GPANO_XMP.format(
        w=width, h=height
    ).encode("utf-8")
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    Path(jpeg_path).write_bytes(data[:2] + segment + data[2:])
    return True


# ----------------------------------------------------------------- driver

def stitch_file(job):
    """One file. `job` is a dict so multiprocessing can pickle it."""
    src = Path(job["src"])
    dst = Path(job["dst"])
    t0 = time.time()
    rgb, exif = load_image(src)
    if not looks_like_dual_fisheye(rgb):
        return (str(src), "SKIPPED",
                "does not look like a dual-fisheye file (corners are not "
                "black) — it may already be stitched; skipped")
    out_w = job["width"] or rgb.shape[1]
    pano = stitch_array(rgb, out_w, job["fov"], job["yaw"],
                        layout=job["layout"],
                        radius_scale=job["radius_scale"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_kw = {"quality": job["quality"]}
    if exif:
        save_kw["exif"] = exif
    Image.fromarray(pano).save(dst, **save_kw)
    xmp = ""
    if job["xmp"]:
        xmp = " + GPano XMP" if inject_gpano_xmp(dst, out_w, out_w // 2) else " (XMP failed)"
    dt = time.time() - t0
    return (str(src), str(dst),
            f"{rgb.shape[1]}x{rgb.shape[0]} -> {out_w}x{out_w // 2} "
            f"in {dt:.1f}s{xmp}")


def collect_inputs(inputs):
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*")
                            if f.is_file() and f.suffix.lower() in SUPPORTED)
        elif p.is_file():
            files.append(p)
        else:
            # a glob pattern
            files += sorted(Path(".").glob(item))
    # de-dup, keep order
    seen, out = set(), []
    for f in files:
        if str(f) not in seen:
            seen.add(str(f))
            out.append(f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Stitch Insta360 X3 dual-fisheye photos (.INSP/.DNG/.JPG) "
                    "into equirectangular panoramas.")
    ap.add_argument("inputs", nargs="+",
                    help="photo file(s), directory, or glob")
    ap.add_argument("--output-dir", "-o", default=None,
                    help="output folder (default: <input>_pano next to each file, "
                         "or ./panos for a directory input)")
    ap.add_argument("--width", type=int, default=0,
                    help="output width in px (default: source width, e.g. 5952); "
                         "use 2880 for a fast preview")
    ap.add_argument("--fov", type=float, default=DEFAULT_FOV,
                    help=f"fisheye FOV in degrees (default {DEFAULT_FOV}; "
                         "try 195/205 if the seam ghosts)")
    ap.add_argument("--yaw", type=float, default=0,
                    help="rotate panorama, degrees (180 = face the other way)")
    ap.add_argument("--layout", default="auto",
                    choices=["auto", "side-by-side", "top-bottom"])
    ap.add_argument("--radius-scale", type=float, default=1.0,
                    help="scale the fisheye circle radius (default 1.0)")
    ap.add_argument("--quality", type=int, default=92,
                    help="JPEG quality (default 92)")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="parallel files (default 1)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-xmp", dest="xmp", action="store_false",
                    help="skip GPano XMP panorama tag")
    args = ap.parse_args(argv)

    files = collect_inputs(args.inputs)
    if not files:
        print("no supported files found (.insp/.dng/.jpg)", file=sys.stderr)
        return 2

    jobs, skipped = [], 0
    for src in files:
        stem = src.stem
        if src.suffix.lower() == ".dng":
            stem += "_dng"
        out_dir = (Path(args.output_dir) if args.output_dir
                   else (src.parent / "panos" if len(files) > 1 or src.parent != Path(".")
                         else Path("panos")))
        dst = out_dir / f"{stem}_pano.jpg"
        if dst.exists() and not args.overwrite:
            print(f"skip (exists, use --overwrite): {dst}")
            skipped += 1
            continue
        jobs.append({"src": str(src), "dst": str(dst), "width": args.width,
                     "fov": args.fov, "yaw": args.yaw, "layout": args.layout,
                     "radius_scale": args.radius_scale,
                     "quality": args.quality, "xmp": args.xmp})

    if not jobs:
        print(f"nothing to do ({skipped} already exist)")
        return 0

    print(f"stitching {len(jobs)} file(s), FOV={args.fov} yaw={args.yaw} "
          f"jobs={args.jobs}")
    ok = fail = 0
    if args.jobs > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        with Pool(min(args.jobs, len(jobs))) as pool:
            results = pool.map(stitch_file, jobs)
    else:
        results = [stitch_file(j) for j in jobs]
    for src, dst, msg in results:
        if dst == "SKIPPED":
            skipped += 1
            print(f"SKIP {src}: {msg}")
        elif dst:
            ok += 1
            print(f"OK  {src}\n    -> {dst}  ({msg})")
        else:
            fail += 1
            print(f"FAIL {src}: {msg}", file=sys.stderr)
    print(f"done: {ok} stitched, {fail} failed, {skipped} skipped")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
