JRYBeats is an extra-simplified Python music app built with Pygame.

## Includes

- DRUMS tab: 16-step drum sequencer
- Synthesized kick, snare, hi-hat, clap, and percussion
- PIANO tab: piano roll with SOFT, PLUCK, BASS, and KEYS
- Piano roll editing is per selected instrument; other instruments in the same cell appear as smaller stacked color bars
- AUDIO tab: audio import, waveform display, and microphone recording
- MIXER tab: volume, pan, mute, and solo for drums, each piano instrument, and audio tracks
- Mixer strips scroll horizontally when there are more tracks than fit on screen
- SONG tab: arrange playback from drum, piano, and audio patterns `00`–`99` (empty slots show `--`)
- Playback follows the SONG order, 16 steps per row, then loops
- Project save/load using `.jry` files, including all patterns, the SONG order, and mixer state
- No external image or drum-sample assets are required

## Run locally

Install Python 3.11+ and then run:

```bash
pip install -r requirements.txt
python JRYBeats.py
```

### macOS

Install SDL, the mixer library, and Tk:

```bash
brew install sdl2 sdl2_mixer python-tk
```

Then install pygame-ce (Community Edition) plus the remaining Python packages:

```bash
pip install pygame-ce numpy sounddevice
python JRYBeats.py
```

`pygame-ce` works more reliably with current Homebrew Python than classic `pygame`. Do not also install `pygame` from `requirements.txt` on Mac, because it can replace pygame-ce. Save, Load, and Import use the native macOS file dialog.

## Notes

Changes from YT Video : The drum icons are drawn in Pygame. The clap is synthesized in code. The snare and hi-hat use differently EQ-shaped noise so they have distinct frequencies.

## License

JRYBeats is licensed under the MIT License. See the `LICENSE` file for details.
