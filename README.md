# insta360-stitch

Stitch Insta360 X3 photos into real 360° panoramas with plain Python
(numpy + Pillow, no Insta360 Studio, no OpenCV).

An X3 `.INSP` file is two circular fisheye images side by side — it is
not a panorama yet. This re-projects both circles onto a sphere and
unwraps it as a 2:1 equirectangular JPEG with GPano XMP, so 360 viewers
(Google Photos, Facebook, etc.) recognise it.

## Install

```bash
pip install -r requirements.txt
# for .DNG files only:
pip install rawpy
```

## Use

```bash
# one photo — panorama lands in ./panos/ next to it
python3 stitch.py IMG_20261109_174210_00_005.insp

# a whole folder, 4 at a time
python3 stitch.py ~/photos/x3/ --output-dir ~/photos/panos --jobs 4

# fast preview first (2880 wide instead of 5952)
python3 stitch.py photo.insp --width 2880

# if the seam ghosts, try the other FOV; 180 flips which way it faces
python3 stitch.py photo.insp --fov 195
python3 stitch.py photo.insp --yaw 180
```

Output is `<name>_pano.jpg`, same width as the source by default
(5952×2976 for 18 MP, 11968×5984 for 72 MP).

## How it works

* Layout auto-detected from size: side-by-side, each circle diameter =
  image height, centres `(H/2, H/2)` and `(3H/2, H/2)`.
* Equidistant fisheye: `r = R · θ / (FOV/2)`, FOV default 200°.
  Insta360 quotes 195° per lens; on the sample photo 200° gave the
  cleanest seam (190/195/200/205/210 were compared — at 195 the door
  plaque ghosts one way, at 205 the other, at 200 it is single).
* Each output pixel's direction is looked up in the front lens
  (left circle) and the back lens (right circle, x flipped because it
  faces the opposite way), bilinear-sampled, and feather-blended
  across the overlap band.

## Limits

* Geometric stitch only — no optical flow like Insta360 Studio.
  Subjects closer than ~0.6 m (Insta360's safe stitching distance)
  can ghost/double at the seam because of parallax between the two
  lenses. The selfie stick at the nadir may smear at the very bottom.
* DNG (`rawpy`) and 72 MP paths are implemented from the file layout
  but were **not** verified on a real X3 DNG — only the 18 MP `.INSP`
  sample was run end-to-end. If a DNG fails, paste the error back.
* Files that are already stitched (corners not black) are skipped
  with a message instead of being re-stitched.
