# Orbital Fireflies: 68 years of filling up the sky

*From one blinking dot in 1957 to 18,592 in 2025.*

Earth spinning in the dark, and every object we've left in orbit since 1957 glowing around it like a firefly.

It starts with one dot, Sputnik 1, in October 1957. By the end of 2025 there are 18,592 of them and you can barely see the planet through the swarm. I wanted something that shows how crowded it's gotten up there without needing a single chart.

Square video (1080x1080), 42 seconds, made for posting on Instagram etc.

## Where the numbers come from

The count on screen isn't made up. It's the yearly number of objects still in orbit from Our World in Data, who take it from the U.S. Space Force's Space-Track catalogue. It's split into LEO, MEO, GEO and high orbits, and the video uses each split separately.

A few numbers from the data, if you're curious:

| year | objects in orbit |
|------|-----------------:|
| 1958 | 2 |
| 1970 | 600 |
| 1990 | 3,047 |
| 2010 | 5,408 |
| 2019 | 7,221 |
| 2025 | 18,592 |

Look at the last six years. It more than doubled after 2019, which is basically Starlink.

One thing to be clear about: this counts payloads **and** old rocket bodies, not just working satellites. It does **not** include debris fragments, so the real picture is worse than this.

## What's real and what isn't

- **Real:** how many dots there are at each point in time, and the physics of how they move (Kepler's laws, so low orbits zip around and the geostationary ring stays put over the same spot on Earth).
- **Not real:** where each individual dot is. Nobody has orbit data for every object back to the 60s, so each one gets a typical orbit for its type: GPS-like for MEO, Starlink shells after 2019, Molniya orbits for the high ones, and so on.
- **Squished:** altitudes. At true scale, LEO is a thin film on the surface and GEO is way out past the edge of the frame. I compress the distance from Earth's centre on a log curve so both fit. Angles and speeds aren't touched.
- Time is sped up a lot, obviously. One second of video is about 21 minutes of orbit, while the calendar races through 68 years.

## Running it

You need Python 3.10+ and ffmpeg on your PATH. No GPU, it all runs on the CPU.

```
pip install numpy opencv-python pillow
python space_fireflies.py
```

The video lands in `output/space_fireflies_1957_2025_1x1.mp4`. The full render took me about 5-6 minutes.

If you prefer Jupyter, open `orbital_fireflies.ipynb` instead. It's the same code split into sections with notes. Change the settings cell near the bottom and run everything. It saves a few preview frames by default. Set `stills=""` to render the full video.

Some options:

```
python space_fireflies.py --size 720             # smaller, quicker test
python space_fireflies.py --stills 45,600,1250   # just save a few frames as PNGs
python space_fireflies.py --start 0 --end 300    # render a chunk (handy if your terminal times out)
python space_fireflies.py --crf 22               # smaller file, a bit lower quality
```

If you render in chunks, you can join them afterwards with ffmpeg's concat.

## Files

```
space_fireflies.py                 the whole thing, one script
orbital_fireflies.ipynb            same code as a notebook
data/satellite_history_clean.csv   year, LEO, MEO, GEO, HEO, total
assets/earth_texture.jpg           NASA Blue Marble (day side)
assets/night_lights.jpg            NASA city lights (night side)
```

The stars are generated in code, so there's no background image to download.

## Want to tweak it?

Everything is near the top of the script:

- `REGIME_COLOR` sets the firefly colours
- `INTRO_S`, `GROW1_S`, `GROW2_S`, `HOLD_S` set how long each part lasts
- `CAPTIONS` holds the text that pops up along the way
- `camera_distance()` controls the zoom out

## Credits

- Object counts: [Our World in Data](https://ourworldindata.org/grapher/space-objects-by-orbit), based on U.S. Space Force / Space-Track data (CC BY 4.0)
- Earth textures: NASA Blue Marble and NASA Earth at Night (public domain)

If you use the video somewhere, please keep the data credit line at the bottom of the frame.
