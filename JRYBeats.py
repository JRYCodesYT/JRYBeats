import json
import os
import subprocess
import sys
import time
import wave
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pygame
import sounddevice as sd

# App setup
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2)
pygame.mixer.set_num_channels(32)
sample_rate = 44100
WIDTH = 1000
HEIGHT = 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('JRYBeats')
clock = pygame.time.Clock()
BACKGROUND = (245, 245, 242)
TEXT_COLOR = (45, 45, 45)
SECONDARY_TEXT = (120, 120, 120)
LINE_COLOR = (70, 70, 70)
LIGHT_LINE = (205, 205, 200)
BLUE = (35, 85, 170)
PLAYHEAD_BLUE = (75, 135, 220)
GREEN = (40, 150, 65)
GREEN_HOVER = (55, 175, 80)
RED = (210, 60, 60)
PURPLE = (110, 95, 210)
PURPLE_HOVER = (130, 115, 225)
ORANGE = (215, 130, 45)
BUTTON_BACKGROUND = (250, 250, 247)
STEP_BACKGROUND = (252, 252, 250)
STEP_HOVER = (225, 225, 220)
BLACK_KEY = (45, 45, 48)
WHITE_KEY = (245, 245, 242)
title_font = pygame.font.Font(None, 40)
section_font = pygame.font.Font(None, 27)
track_font = pygame.font.Font(None, 25)
small_font = pygame.font.Font(None, 21)
tiny_font = pygame.font.Font(None, 17)
step_font = pygame.font.Font(None, 18)

# Drum synthesis
def make_sound(wave):
    wave = np.clip(wave, -1, 1)
    audio = (wave * 32767).astype(np.int16)
    stereo = np.column_stack((audio, audio))
    stereo = np.ascontiguousarray(stereo)
    return pygame.sndarray.make_sound(stereo)
duration = 0.5
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
frequency = 50 + 210 * np.exp(-35 * t)
phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
kick_wave = np.sin(phase)
kick_wave += 0.18 * np.sin(2 * phase)
kick_wave *= np.exp(-7 * t)
click = np.random.uniform(-1, 1, len(t))
click *= np.exp(-100 * t)
kick_wave += click * 0.12
kick_wave *= 0.85
kick = make_sound(kick_wave)

def eq_noise(noise, low_cut=0, high_cut=None, peak_freq=None, peak_gain=0.0):
    spectrum = np.fft.rfft(noise)
    frequencies = np.fft.rfftfreq(len(noise), 1 / sample_rate)
    shape = np.ones_like(frequencies)
    if low_cut > 0:
        shape *= np.clip(frequencies / low_cut, 0.0, 1.0)
    if high_cut is not None:
        shape *= np.clip(high_cut / np.maximum(frequencies, 1), 0.0, 1.0)
    if peak_freq is not None and peak_gain != 0:
        width = max(1.0, peak_freq * 0.55)
        bell = np.exp(-0.5 * ((frequencies - peak_freq) / width) ** 2)
        shape *= 1.0 + bell * peak_gain
    spectrum *= shape
    filtered = np.fft.irfft(spectrum, n=len(noise))
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered /= peak
    return filtered

# The snare keeps more midrange; the hat is pushed much higher.
snare_duration = 0.32
snare_t = np.linspace(0, snare_duration, int(sample_rate * snare_duration), endpoint=False)
snare_noise = np.random.uniform(-1, 1, len(snare_t))
snare_noise = eq_noise(snare_noise, low_cut=550, high_cut=9500, peak_freq=2400, peak_gain=1.2)
snare_body = np.sin(2 * np.pi * 185 * snare_t)
snare_body += 0.35 * np.sin(2 * np.pi * 330 * snare_t)
noise_envelope = np.exp(-13 * snare_t)
body_envelope = np.exp(-18 * snare_t)
snare_attack = np.random.uniform(-1, 1, len(snare_t))
snare_attack = eq_noise(snare_attack, low_cut=1800, high_cut=12000, peak_freq=4500, peak_gain=0.8)
snare_attack *= np.exp(-90 * snare_t)
snare_wave = snare_noise * noise_envelope * 0.72 + snare_body * body_envelope * 0.38 + snare_attack * 0.2
snare_wave *= 0.75
snare = make_sound(snare_wave)
hihat_duration = 0.1
hihat_t = np.linspace(0, hihat_duration, int(sample_rate * hihat_duration), endpoint=False)
hihat_noise = np.random.uniform(-1, 1, len(hihat_t))
hihat_noise = eq_noise(hihat_noise, low_cut=5500, high_cut=18000, peak_freq=10500, peak_gain=1.5)
hihat_envelope = np.exp(-48 * hihat_t)
hihat_wave = hihat_noise * hihat_envelope * 0.48
for metallic_frequency in (6400, 7900, 10100, 12400):
    hihat_wave += np.sin(2 * np.pi * metallic_frequency * hihat_t) * np.exp(-55 * hihat_t) * 0.025
hihat = make_sound(hihat_wave)
perc_duration = 0.12
perc_t = np.linspace(0, perc_duration, int(sample_rate * perc_duration), endpoint=False)
perc_noise = np.random.uniform(-1, 1, len(perc_t))
perc_wave = perc_noise * np.exp(-28 * perc_t) * 0.35
perc = make_sound(perc_wave)

# Two fast hits followed by a longer noisy tail gives the clap its shape.
clap_duration = 0.42
clap_t = np.linspace(0, clap_duration, int(sample_rate * clap_duration), endpoint=False)
clap_noise = np.random.uniform(-1, 1, len(clap_t))
clap_noise = eq_noise(clap_noise, low_cut=900, high_cut=12500, peak_freq=3200, peak_gain=1.5)
first_start = 0.0
first_decay = 0.008
first_burst = np.where(clap_t >= first_start, np.exp(-(clap_t - first_start) / first_decay), 0)
first_burst *= clap_t < 0.025
second_start = 0.022
second_decay = 0.01
second_burst = np.where(clap_t >= second_start, np.exp(-(clap_t - second_start) / second_decay), 0)
second_burst *= clap_t < second_start + 0.03
tail_start = 0.045
tail = np.where(clap_t >= tail_start, np.exp(-(clap_t - tail_start) / 0.105), 0)
clap_envelope = first_burst * 1.0 + second_burst * 0.95 + tail * 0.55
clap_wave = clap_noise * clap_envelope
clap_body = np.sin(2 * np.pi * 1150 * clap_t) + 0.5 * np.sin(2 * np.pi * 1750 * clap_t)
clap_body *= np.where(clap_t >= tail_start, np.exp(-(clap_t - tail_start) / 0.055), 0)
clap_wave += clap_body * 0.08
second_texture = np.random.uniform(-1, 1, len(clap_t))
second_texture = eq_noise(second_texture, low_cut=1200, high_cut=11000, peak_freq=4000, peak_gain=1.0)
second_texture *= second_burst
clap_wave += second_texture * 0.22
clap_wave *= 0.72
clap = make_sound(clap_wave)
drum_tracks = ['KICK', 'SNARE', 'HI-HAT', 'CLAP', 'PERC']
drum_sounds = [kick, snare, hihat, clap, perc]
drum_mixer = [{'volume': 0.85, 'muted': False, 'solo': False, 'pan': 0.0} for _ in drum_tracks]
mixer_drag = None
mixer_scroll_x = 0
mixer_scrollbar_drag = None


# Mixer helpers
def pan_to_lr(volume, pan):
    pan = max(-1.0, min(1.0, pan))
    volume = max(0.0, min(1.0, volume))
    if pan < 0:
        left = volume
        right = volume * (1.0 + pan)
    else:
        left = volume * (1.0 - pan)
        right = volume
    return (left, right)

def any_track_soloed():
    if any((track['solo'] for track in drum_mixer)):
        return True
    if any((instrument_mixer[name]['solo'] for name in instruments)):
        return True
    return any((track.get('solo', False) for track in all_audio_tracks()))

def mixer_track_audible(track):
    if track.get('muted', False):
        return False
    if any_track_soloed():
        return track.get('solo', False)
    return True

def play_sound_with_mixer(sound, track):
    if not mixer_track_audible(track):
        return None
    channel = sound.play()
    if channel is not None:
        left, right = pan_to_lr(track.get('volume', 1.0), track.get('pan', 0.0))
        channel.set_volume(left, right)
    return channel

def refresh_playing_audio_mixer():
    for track in all_audio_tracks():
        channel = track.get('channel')
        if channel is None:
            continue
        if not channel.get_busy():
            track['channel'] = None
            continue
        if not mixer_track_audible(track):
            channel.stop()
            track['channel'] = None
            continue
        left, right = pan_to_lr(track.get('volume', 1.0), track.get('pan', 0.0))
        channel.set_volume(left, right)


# Simple built-in icons keep the app self-contained.
def make_icon_surface(size=(48, 48)):
    return pygame.Surface(size, pygame.SRCALPHA)

def create_kick_icon():
    surface = make_icon_surface()
    pygame.draw.circle(surface, TEXT_COLOR, (24, 24), 18, 3)
    pygame.draw.circle(surface, SECONDARY_TEXT, (24, 24), 5, 2)
    pygame.draw.line(surface, TEXT_COLOR, (12, 38), (8, 46), 3)
    pygame.draw.line(surface, TEXT_COLOR, (36, 38), (40, 46), 3)
    return surface

def create_snare_icon():
    surface = make_icon_surface()
    pygame.draw.ellipse(surface, TEXT_COLOR, (7, 10, 34, 11), 2)
    pygame.draw.rect(surface, TEXT_COLOR, (7, 15, 34, 20), 2)
    pygame.draw.ellipse(surface, TEXT_COLOR, (7, 29, 34, 11), 2)
    pygame.draw.line(surface, SECONDARY_TEXT, (10, 20), (38, 31), 2)
    pygame.draw.line(surface, SECONDARY_TEXT, (10, 31), (38, 20), 2)
    return surface

def create_hihat_icon():
    surface = make_icon_surface()
    pygame.draw.line(surface, TEXT_COLOR, (24, 12), (24, 42), 3)
    pygame.draw.line(surface, TEXT_COLOR, (10, 19), (38, 19), 3)
    pygame.draw.line(surface, SECONDARY_TEXT, (13, 23), (35, 23), 2)
    pygame.draw.line(surface, TEXT_COLOR, (17, 42), (31, 42), 3)
    return surface

def create_clap_icon():
    surface = make_icon_surface()
    pygame.draw.polygon(surface, TEXT_COLOR, [(8, 28), (14, 15), (18, 17), (16, 27), (22, 13), (26, 15), (22, 30), (29, 18), (33, 21), (27, 35), (17, 39)], 2)
    pygame.draw.polygon(surface, SECONDARY_TEXT, [(40, 26), (35, 14), (31, 17), (33, 27), (27, 13), (24, 16), (29, 31), (22, 20), (19, 23), (25, 37), (35, 39)], 2)
    return surface

def create_perc_icon():
    surface = make_icon_surface()
    pygame.draw.ellipse(surface, TEXT_COLOR, (9, 5, 24, 28), 3)
    pygame.draw.line(surface, TEXT_COLOR, (27, 29), (39, 44), 5)
    pygame.draw.circle(surface, SECONDARY_TEXT, (19, 16), 3)
    pygame.draw.circle(surface, SECONDARY_TEXT, (25, 22), 3)
    return surface

def create_microphone_icon():
    surface = pygame.Surface((24, 24), pygame.SRCALPHA)
    pygame.draw.rect(surface, TEXT_COLOR, (8, 2, 8, 13), border_radius=4)
    pygame.draw.arc(surface, TEXT_COLOR, (5, 7, 14, 11), 3.14159, 6.28318, 2)
    pygame.draw.line(surface, TEXT_COLOR, (12, 17), (12, 22), 2)
    pygame.draw.line(surface, TEXT_COLOR, (8, 22), (16, 22), 2)
    return surface
kick_image = create_kick_icon()
snare_image = create_snare_icon()
hihat_image = create_hihat_icon()
clap_image = create_clap_icon()
perc_image = create_perc_icon()
microphone_image = create_microphone_icon()
drum_images = [kick_image, snare_image, hihat_image, clap_image, perc_image]
num_steps = 16
NUM_PATTERNS = 100
step_size = 28
step_gap = 8
sequencer_start_x = 300
track_start_y = 235
row_height = 68

def make_empty_drum_pattern():
    return [[False for _ in range(num_steps)] for _ in drum_tracks]

drum_patterns = [make_empty_drum_pattern() for _ in range(NUM_PATTERNS)]
edit_drum_pattern = 0
pattern = drum_patterns[0]
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def note_to_midi(note):
    if '#' in note:
        note_name = note[:2]
        octave = int(note[2:])
    else:
        note_name = note[0]
        octave = int(note[1:])
    return 12 * (octave + 1) + NOTE_NAMES.index(note_name)

def note_frequency(note):
    midi = note_to_midi(note)
    return 440 * 2 ** ((midi - 69) / 12)
piano_notes = []
for midi in range(note_to_midi('C3'), note_to_midi('C6') + 1):
    octave = midi // 12 - 1
    name = NOTE_NAMES[midi % 12]
    piano_notes.append(f'{name}{octave}')
piano_notes.reverse()
instruments = ['SOFT', 'PLUCK', 'BASS', 'KEYS']
melody_instrument = 'SOFT'
melody_volume = 0.7
instrument_mixer = {name: {'volume': 0.7, 'muted': False, 'solo': False, 'pan': 0.0} for name in instruments}
INSTRUMENT_COLORS = {'SOFT': (110, 95, 210), 'PLUCK': (45, 155, 95), 'BASS': (215, 130, 45), 'KEYS': (45, 125, 190)}
INSTRUMENT_HOVER_COLORS = {'SOFT': (130, 115, 225), 'PLUCK': (65, 175, 115), 'BASS': (230, 150, 65), 'KEYS': (65, 145, 210)}

def create_synth_wave(note, instrument):
    freq = note_frequency(note)
    duration = 0.4
    if instrument == 'PLUCK':
        duration = 0.25
    elif instrument == 'BASS':
        freq /= 2
        duration = 0.45
    elif instrument == 'KEYS':
        duration = 0.5
    count = int(sample_rate * duration)
    note_t = np.linspace(0, duration, count, endpoint=False)
    if instrument == 'SOFT':
        note_wave = np.sin(2 * np.pi * freq * note_t) * 0.75
        note_wave += np.sin(2 * np.pi * freq * 2 * note_t) * 0.15
        fade = np.exp(-5 * note_t)
    elif instrument == 'PLUCK':
        note_wave = np.sin(2 * np.pi * freq * note_t) * 0.65
        note_wave += np.sin(2 * np.pi * freq * 2 * note_t) * 0.25
        fade = np.exp(-14 * note_t)
    elif instrument == 'BASS':
        note_wave = np.sin(2 * np.pi * freq * note_t) * 0.8
        note_wave += np.sin(2 * np.pi * freq * 2 * note_t) * 0.12
        fade = np.exp(-6 * note_t)
    else:
        note_wave = np.sin(2 * np.pi * freq * note_t) * 0.55
        note_wave += np.sin(2 * np.pi * freq * 2 * note_t) * 0.22
        note_wave += np.sin(2 * np.pi * freq * 3 * note_t) * 0.1
        fade = np.exp(-4 * note_t)
    attack = np.minimum(1, note_t / 0.01)
    note_wave = note_wave * fade * attack * 0.65
    return note_wave
melody_sounds = {}
for instrument in instruments:
    for note in piano_notes:
        melody_sounds[note, instrument] = make_sound(create_synth_wave(note, instrument))
def make_empty_piano_pattern():
    return [[[] for _ in range(num_steps)] for _ in piano_notes]

piano_patterns = [make_empty_piano_pattern() for _ in range(NUM_PATTERNS)]
edit_piano_pattern = 0
melody_pattern = piano_patterns[0]

def piano_cell_instruments(cell):
    if isinstance(cell, list):
        return [item for item in instruments if item in cell]
    if cell in instruments:
        return [cell]
    if cell is True:
        return [melody_instrument]
    return []

def toggle_selected_piano_cell(note_index, step):
    present = piano_cell_instruments(melody_pattern[note_index][step])
    if melody_instrument in present:
        present = [item for item in present if item != melody_instrument]
        added = False
    else:
        present.append(melody_instrument)
        present = [item for item in instruments if item in present]
        added = True
    melody_pattern[note_index][step] = present
    return added

def draw_piano_cell(cell_rect, cell, mouse_over, sharp):
    present = piano_cell_instruments(cell)
    others = [item for item in present if item != melody_instrument]
    has_selected = melody_instrument in present
    if sharp:
        base_color = (232, 232, 228)
    else:
        base_color = (248, 248, 245)
    if mouse_over and not has_selected:
        base_color = STEP_HOVER
    pygame.draw.rect(screen, base_color, cell_rect)
    other_bar_height = 6
    stack_height = other_bar_height * len(others)
    selected_height = cell_rect.height - stack_height
    if has_selected:
        selected_color = INSTRUMENT_HOVER_COLORS[melody_instrument] if mouse_over else INSTRUMENT_COLORS[melody_instrument]
        pygame.draw.rect(screen, selected_color, pygame.Rect(cell_rect.x, cell_rect.y, cell_rect.width, selected_height))
    for i, other in enumerate(others):
        y = cell_rect.y + selected_height + i * other_bar_height
        pygame.draw.rect(screen, INSTRUMENT_COLORS[other], pygame.Rect(cell_rect.x, y, cell_rect.width, other_bar_height))
    pygame.draw.rect(screen, LIGHT_LINE, cell_rect, 1)
piano_grid_start_x = 130
piano_step_width = 45
piano_grid_top = 225
piano_row_height = 28
visible_piano_rows = 12
piano_scroll = 11
audio_patterns = [[] for _ in range(NUM_PATTERNS)]
edit_audio_pattern = 0
audio_tracks = audio_patterns[0]
song_order = [{'drums': 0, 'piano': 0, 'audio': None}]
current_order_index = 0
song_scroll = 0
SONG_LIST_TOP = 248
SONG_ROW_HEIGHT = 38
SONG_VISIBLE_ROWS = 9

def all_audio_tracks():
    tracks = []
    for pattern_tracks in audio_patterns:
        tracks.extend(pattern_tracks)
    return tracks

def bind_edit_patterns():
    global pattern
    global melody_pattern
    global audio_tracks
    pattern = drum_patterns[edit_drum_pattern]
    melody_pattern = piano_patterns[edit_piano_pattern]
    audio_tracks = audio_patterns[edit_audio_pattern]

def set_edit_drum_pattern(index):
    global edit_drum_pattern
    edit_drum_pattern = clamp_pattern_index(index, 0)
    bind_edit_patterns()

def set_edit_piano_pattern(index):
    global edit_piano_pattern
    edit_piano_pattern = clamp_pattern_index(index, 0)
    bind_edit_patterns()

def set_edit_audio_pattern(index):
    global edit_audio_pattern
    edit_audio_pattern = clamp_pattern_index(index, 0)
    bind_edit_patterns()

def format_pattern_id(value):
    if value is None:
        return '--'
    return f'{int(value):02d}'

def cycle_pattern_slot(value, delta):
    if value is None:
        if delta > 0:
            return 0
        return NUM_PATTERNS - 1
    next_value = int(value) + delta
    if next_value < 0 or next_value >= NUM_PATTERNS:
        return None
    return next_value

def clamp_song_scroll():
    global song_scroll
    max_scroll = max(0, len(song_order) - SONG_VISIBLE_ROWS)
    song_scroll = max(0, min(song_scroll, max_scroll))

def active_song_row():
    if not song_order:
        return {'drums': edit_drum_pattern, 'piano': edit_piano_pattern, 'audio': edit_audio_pattern}
    return song_order[current_order_index % len(song_order)]

def empty_song_row():
    return {'drums': None, 'piano': None, 'audio': None}

def pattern_selector_rects(x, y):
    minus_rect = pygame.Rect(x, y, 28, 26)
    value_rect = pygame.Rect(x + 30, y, 42, 26)
    plus_rect = pygame.Rect(x + 74, y, 28, 26)
    return (minus_rect, value_rect, plus_rect)

def draw_pattern_selector(x, y, value, allow_empty=False):
    minus_rect, value_rect, plus_rect = pattern_selector_rects(x, y)
    pygame.draw.rect(screen, BUTTON_BACKGROUND, minus_rect)
    pygame.draw.rect(screen, LINE_COLOR, minus_rect, 1)
    pygame.draw.rect(screen, BUTTON_BACKGROUND, value_rect)
    pygame.draw.rect(screen, LINE_COLOR, value_rect, 1)
    pygame.draw.rect(screen, BUTTON_BACKGROUND, plus_rect)
    pygame.draw.rect(screen, LINE_COLOR, plus_rect, 1)
    minus_text = tiny_font.render('<', True, TEXT_COLOR)
    plus_text = tiny_font.render('>', True, TEXT_COLOR)
    if allow_empty:
        label = format_pattern_id(value)
    else:
        label = f'{int(value):02d}'
    value_text = tiny_font.render(label, True, TEXT_COLOR)
    screen.blit(minus_text, minus_text.get_rect(center=minus_rect.center))
    screen.blit(value_text, value_text.get_rect(center=value_rect.center))
    screen.blit(plus_text, plus_text.get_rect(center=plus_rect.center))
    return (minus_rect, plus_rect)

def song_row_control_rects(visible_index):
    y = SONG_LIST_TOP + visible_index * SONG_ROW_HEIGHT
    drums_minus, drums_value, drums_plus = pattern_selector_rects(118, y)
    piano_minus, piano_value, piano_plus = pattern_selector_rects(318, y)
    audio_minus, audio_value, audio_plus = pattern_selector_rects(518, y)
    delete_rect = pygame.Rect(730, y, 28, 26)
    return {
        'y': y,
        'drums_minus': drums_minus,
        'drums_plus': drums_plus,
        'piano_minus': piano_minus,
        'piano_plus': piano_plus,
        'audio_minus': audio_minus,
        'audio_plus': audio_plus,
        'delete': delete_rect,
        'row': pygame.Rect(20, y - 4, 760, SONG_ROW_HEIGHT),
    }
supported_audio_extensions = ['.wav', '.mp3', '.ogg']
MIXER_STRIP_WIDTH = 105
MIXER_STRIP_INNER_WIDTH = 96
MIXER_STRIP_START_X = 20
MIXER_STRIP_TOP = 225
MIXER_STRIP_HEIGHT = 390
MIXER_VIEW_LEFT = 20
MIXER_VIEW_RIGHT = WIDTH - 20
mixer_scroll_left_rect = pygame.Rect(WIDTH - 90, 198, 32, 26)
mixer_scroll_right_rect = pygame.Rect(WIDTH - 52, 198, 32, 26)

def get_mixer_tracks():
    tracks = []
    for i, name in enumerate(drum_tracks):
        tracks.append((name, drum_mixer[i]))
    for name in instruments:
        tracks.append((name, instrument_mixer[name]))
    for audio_track in audio_tracks:
        tracks.append((audio_track.get('name', 'AUDIO')[:10], audio_track))
    return tracks

def mixer_visible_width():
    return MIXER_VIEW_RIGHT - MIXER_STRIP_START_X

def mixer_content_width():
    return len(get_mixer_tracks()) * MIXER_STRIP_WIDTH

def mixer_max_scroll():
    return max(0, mixer_content_width() - mixer_visible_width())

def clamp_mixer_scroll():
    global mixer_scroll_x
    mixer_scroll_x = max(0, min(int(mixer_scroll_x), mixer_max_scroll()))

def mixer_strip_screen_x(index):
    return MIXER_STRIP_START_X + index * MIXER_STRIP_WIDTH - mixer_scroll_x

def mixer_scrollbar_geometry():
    bar = pygame.Rect(MIXER_VIEW_LEFT, HEIGHT - 24, MIXER_VIEW_RIGHT - MIXER_VIEW_LEFT, 12)
    content = mixer_content_width()
    visible = mixer_visible_width()
    if content <= visible:
        return (bar, None)
    thumb_width = max(36, int(bar.width * visible / content))
    max_scroll = mixer_max_scroll()
    if max_scroll <= 0:
        thumb_x = bar.x
    else:
        thumb_x = bar.x + int((bar.width - thumb_width) * (mixer_scroll_x / max_scroll))
    thumb = pygame.Rect(thumb_x, bar.y, thumb_width, bar.height)
    return (bar, thumb)

def mixer_control_rects(strip_x):
    mute_rect = pygame.Rect(strip_x + 12, MIXER_STRIP_TOP + 28, 34, 26)
    solo_rect = pygame.Rect(strip_x + 54, MIXER_STRIP_TOP + 28, 34, 26)
    volume_rect = pygame.Rect(strip_x + 45, MIXER_STRIP_TOP + 82, 14, 230)
    pan_rect = pygame.Rect(strip_x + 12, MIXER_STRIP_TOP + 345, 76, 18)
    return (mute_rect, solo_rect, volume_rect, pan_rect)

def handle_mixer_click(pos):
    global mixer_drag
    global mixer_scroll_x
    global mixer_scrollbar_drag
    if mixer_scroll_left_rect.collidepoint(pos):
        mixer_scroll_x -= MIXER_STRIP_WIDTH
        clamp_mixer_scroll()
        return
    if mixer_scroll_right_rect.collidepoint(pos):
        mixer_scroll_x += MIXER_STRIP_WIDTH
        clamp_mixer_scroll()
        return
    bar, thumb = mixer_scrollbar_geometry()
    if bar.collidepoint(pos):
        if thumb is not None and thumb.collidepoint(pos):
            mixer_scrollbar_drag = pos[0] - thumb.x
        elif thumb is not None:
            ratio = (pos[0] - bar.x - thumb.width / 2) / max(1, bar.width - thumb.width)
            mixer_scroll_x = ratio * mixer_max_scroll()
            clamp_mixer_scroll()
        return
    for i, (name, track_state) in enumerate(get_mixer_tracks()):
        x = mixer_strip_screen_x(i)
        if x + MIXER_STRIP_INNER_WIDTH < MIXER_STRIP_START_X or x > MIXER_VIEW_RIGHT:
            continue
        mute_rect, solo_rect, volume_rect, pan_rect = mixer_control_rects(x)
        if mute_rect.collidepoint(pos):
            track_state['muted'] = not track_state.get('muted', False)
            return
        if solo_rect.collidepoint(pos):
            track_state['solo'] = not track_state.get('solo', False)
            return
        if volume_rect.inflate(18, 0).collidepoint(pos):
            mixer_drag = (track_state, 'volume', volume_rect)
            ratio = (volume_rect.bottom - pos[1]) / volume_rect.height
            track_state['volume'] = max(0.0, min(1.0, ratio))
            return
        if pan_rect.inflate(0, 10).collidepoint(pos):
            mixer_drag = (track_state, 'pan', pan_rect)
            ratio = (pos[0] - pan_rect.left) / pan_rect.width
            track_state['pan'] = max(-1.0, min(1.0, ratio * 2.0 - 1.0))
            return
audio_timeline_x = 210
audio_timeline_width = 740
audio_track_top = 270
audio_track_height = 72
recording_microphone = False
microphone_chunks = []
microphone_stream = None
microphone_record_samplerate = sample_rate
microphone_devices = [(index, device['name']) for index, device in enumerate(sd.query_devices()) if device['max_input_channels'] > 0]
selected_microphone_position = 0
try:
    default_input_device = sd.default.device[0]
    for position, (device_index, device_name) in enumerate(microphone_devices):
        if device_index == default_input_device:
            selected_microphone_position = position
            break
except Exception:
    pass

def get_selected_microphone():
    if not microphone_devices:
        return (None, 'NO MICROPHONE FOUND')
    return microphone_devices[selected_microphone_position]

def import_audio_file(path, destination=None):
    if destination is None:
        destination = audio_tracks
    extension = os.path.splitext(path)[1].lower()
    if extension not in supported_audio_extensions:
        print('Unsupported audio file:', path)
        return
    try:
        sound = pygame.mixer.Sound(path)
        sound_array = pygame.sndarray.array(sound)
        if len(sound_array.shape) == 2:
            waveform = np.mean(sound_array, axis=1)
        else:
            waveform = sound_array
        waveform = waveform.astype(np.float32)
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform /= peak
        destination.append({'name': os.path.basename(path), 'path': path, 'sound': sound, 'waveform': waveform, 'length': sound.get_length(), 'start_step': 0, 'muted': False, 'solo': False, 'pan': 0.0, 'volume': 0.8})
        print('Imported:', os.path.basename(path))
    except Exception as error:
        print("Couldn't import:", path)
        print(error)


# Project files
# macOS: pygame/SDL and Tk both want the Cocoa NSApplication singleton, so creating
# a Tk() root after pygame.init() (and destroying it after each dialog) crashes or
# hangs the file picker. Use the native dialog instead, and keep a single Tk root
# on other platforms.
_dialog_root = None
PROJECT_FILETYPES = [('JRYBeats Project', '*.jry'), ('JSON', '*.json'), ('All Files', '*.*')]
AUDIO_FILETYPES = [('Audio Files', '*.wav *.mp3 *.ogg'), ('WAV', '*.wav'), ('MP3', '*.mp3'), ('OGG', '*.ogg'), ('All Files', '*.*')]

def _escape_applescript(text):
    return str(text).replace('\\', '\\\\').replace('"', '\\"')

def _clear_dialog_mouse_events():
    pygame.event.clear((pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION))

def _macos_file_dialog(mode, title, default_name=''):
    escaped_title = _escape_applescript(title)
    if mode == 'save':
        escaped_name = _escape_applescript(default_name or 'untitled.jry')
        script = (
            f'try\n'
            f'    set theFile to choose file name with prompt "{escaped_title}" default name "{escaped_name}"\n'
            f'    return POSIX path of theFile\n'
            f'on error\n'
            f'    return ""\n'
            f'end try\n'
        )
    else:
        script = (
            f'try\n'
            f'    set theFile to choose file with prompt "{escaped_title}"\n'
            f'    return POSIX path of theFile\n'
            f'on error\n'
            f'    return ""\n'
            f'end try\n'
        )
    try:
        result = subprocess.run(['osascript'], input=script, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()

def _get_dialog_root():
    global _dialog_root
    if _dialog_root is None:
        _dialog_root = tk.Tk()
        _dialog_root.withdraw()
        _dialog_root.attributes('-topmost', True)
    try:
        _dialog_root.lift()
        _dialog_root.focus_force()
        _dialog_root.update()
    except tk.TclError:
        _dialog_root = tk.Tk()
        _dialog_root.withdraw()
        _dialog_root.attributes('-topmost', True)
        _dialog_root.update()
    return _dialog_root

def _tk_file_dialog(mode, title, filetypes, default_name=''):
    root = _get_dialog_root()
    if mode == 'save':
        path = filedialog.asksaveasfilename(title=title, defaultextension='.jry', filetypes=filetypes, initialfile=default_name)
    else:
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    try:
        root.update()
    except tk.TclError:
        pass
    return path or ''

def ask_open_path(title, filetypes):
    if sys.platform == 'darwin':
        path = _macos_file_dialog('open', title)
        _clear_dialog_mouse_events()
        if path is None:
            print("Couldn't open the macOS file dialog.")
            return ''
        return path
    path = _tk_file_dialog('open', title, filetypes)
    _clear_dialog_mouse_events()
    return path

def ask_save_path(title, filetypes, default_name='untitled.jry'):
    if sys.platform == 'darwin':
        path = _macos_file_dialog('save', title, default_name)
        _clear_dialog_mouse_events()
        if path is None:
            print("Couldn't open the macOS file dialog.")
            return ''
        if path and os.path.splitext(path)[1] == '':
            path += '.jry'
        return path
    path = _tk_file_dialog('save', title, filetypes, default_name)
    _clear_dialog_mouse_events()
    return path

def serialize_audio_track(track):
    return {
        'name': track.get('name', 'Audio'),
        'path': os.path.abspath(track.get('path', '')),
        'start_step': track.get('start_step', 0),
        'muted': track.get('muted', False),
        'solo': track.get('solo', False),
        'pan': track.get('pan', 0.0),
        'volume': track.get('volume', 0.8),
    }

def parse_pattern_index(key):
    try:
        index = int(key)
    except (TypeError, ValueError):
        return None
    if 0 <= index < NUM_PATTERNS:
        return index
    return None

def clamp_pattern_index(value, default=0):
    parsed = parse_pattern_index(value)
    if parsed is None:
        return default
    return parsed

def json_ready(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, 'item'):
        return json_ready(value.item())
    return value

def serialize_song_order():
    rows = []
    for row in song_order:
        rows.append({
            'drums': parse_pattern_index(row.get('drums')) if row.get('drums') is not None else None,
            'piano': parse_pattern_index(row.get('piano')) if row.get('piano') is not None else None,
            'audio': parse_pattern_index(row.get('audio')) if row.get('audio') is not None else None,
        })
    return rows

def referenced_pattern_indexes(lane, edit_index):
    used = {clamp_pattern_index(edit_index, 0)}
    for row in song_order:
        parsed = parse_pattern_index(row.get(lane)) if row.get(lane) is not None else None
        if parsed is not None:
            used.add(parsed)
    return used

def apply_saved_audio_track(saved_track, destination, missing_audio):
    audio_path = saved_track.get('path', '')
    if not os.path.exists(audio_path):
        missing_audio.append(audio_path)
        return
    before_count = len(destination)
    import_audio_file(audio_path, destination)
    if len(destination) > before_count:
        loaded_track = destination[-1]
        loaded_track['start_step'] = max(0, min(num_steps - 1, int(saved_track.get('start_step', 0))))
        loaded_track['muted'] = bool(saved_track.get('muted', False))
        loaded_track['solo'] = bool(saved_track.get('solo', False))
        loaded_track['pan'] = max(-1.0, min(1.0, float(saved_track.get('pan', 0.0))))
        loaded_track['volume'] = max(0.0, min(1.0, float(saved_track.get('volume', 0.8))))

def reset_pattern_banks():
    global drum_patterns
    global piano_patterns
    global audio_patterns
    drum_patterns = [make_empty_drum_pattern() for _ in range(NUM_PATTERNS)]
    piano_patterns = [make_empty_piano_pattern() for _ in range(NUM_PATTERNS)]
    audio_patterns = [[] for _ in range(NUM_PATTERNS)]
    bind_edit_patterns()

def save_project():
    if recording_microphone:
        print('Stop microphone recording before saving.')
        return
    path = ask_save_path('Save JRYBeats Project', PROJECT_FILETYPES)
    if not path:
        return
    used_drums = referenced_pattern_indexes('drums', edit_drum_pattern)
    used_piano = referenced_pattern_indexes('piano', edit_piano_pattern)
    used_audio = referenced_pattern_indexes('audio', edit_audio_pattern)
    saved_drums = {}
    saved_piano = {}
    saved_audio = {}
    for index in range(NUM_PATTERNS):
        drum_pat = drum_patterns[index]
        if index in used_drums or any((any(row) for row in drum_pat)):
            saved_drums[f'{index:02d}'] = drum_pat
        piano_pat = piano_patterns[index]
        if index in used_piano or any((any(cell for cell in row) for row in piano_pat)):
            saved_piano[f'{index:02d}'] = piano_pat
        if index in used_audio or audio_patterns[index]:
            saved_audio[f'{index:02d}'] = [serialize_audio_track(track) for track in audio_patterns[index]]
    project = {
        'version': 3,
        'bpm': bpm,
        'current_view': current_view,
        'melody_instrument': melody_instrument,
        'piano_scroll': piano_scroll,
        'drum_mixer': drum_mixer,
        'instrument_mixer': instrument_mixer,
        'melody_mixer': instrument_mixer.get(melody_instrument, instrument_mixer['SOFT']),
        'edit_drum_pattern': edit_drum_pattern,
        'edit_piano_pattern': edit_piano_pattern,
        'edit_audio_pattern': edit_audio_pattern,
        'current_order_index': current_order_index,
        'drum_patterns': saved_drums,
        'piano_patterns': saved_piano,
        'audio_patterns': saved_audio,
        'song_order': serialize_song_order(),
        'pattern': drum_patterns[edit_drum_pattern],
        'melody_pattern': piano_patterns[edit_piano_pattern],
        'audio_tracks': [serialize_audio_track(track) for track in audio_patterns[edit_audio_pattern]],
    }
    try:
        with open(path, 'w', encoding='utf-8') as project_file:
            json.dump(json_ready(project), project_file, indent=2)
        print('Project saved:', path)
    except Exception as error:
        print("Couldn't save project:", path)
        print(error)

def load_project():
    global bpm
    global current_view
    global pattern
    global melody_pattern
    global melody_instrument
    global piano_scroll
    global drum_mixer
    global instrument_mixer
    global dragging_audio
    global mixer_drag
    global drum_patterns
    global piano_patterns
    global audio_patterns
    global audio_tracks
    global edit_drum_pattern
    global edit_piano_pattern
    global edit_audio_pattern
    global song_order
    global current_order_index
    global song_scroll
    global playing
    global current_step
    if recording_microphone:
        print('Stop microphone recording before loading a project.')
        return
    path = ask_open_path('Load JRYBeats Project', PROJECT_FILETYPES)
    if not path:
        return
    try:
        with open(path, 'r', encoding='utf-8') as project_file:
            project = json.load(project_file)
        pygame.mixer.stop()
        playing = False
        current_step = 0
        current_order_index = 0
        bpm = max(min_bpm, min(max_bpm, int(project.get('bpm', bpm))))
        reset_pattern_banks()
        loaded_drum_bank = project.get('drum_patterns')
        if isinstance(loaded_drum_bank, dict):
            for key, saved_pattern in loaded_drum_bank.items():
                index = parse_pattern_index(key)
                if index is None:
                    continue
                if isinstance(saved_pattern, list) and len(saved_pattern) == len(drum_tracks) and all((len(row) == num_steps for row in saved_pattern)):
                    drum_patterns[index] = [[bool(cell) for cell in row] for row in saved_pattern]
        else:
            loaded_pattern = project.get('pattern')
            if isinstance(loaded_pattern, list) and len(loaded_pattern) == len(drum_tracks) and all((len(row) == num_steps for row in loaded_pattern)):
                drum_patterns[0] = [[bool(cell) for cell in row] for row in loaded_pattern]
        saved_instrument = project.get('melody_instrument', melody_instrument)
        if saved_instrument in instruments:
            melody_instrument = saved_instrument
        loaded_piano_bank = project.get('piano_patterns')
        if isinstance(loaded_piano_bank, dict):
            for key, saved_melody in loaded_piano_bank.items():
                index = parse_pattern_index(key)
                if index is None:
                    continue
                if isinstance(saved_melody, list) and len(saved_melody) == len(piano_notes) and all((len(row) == num_steps for row in saved_melody)):
                    piano_patterns[index] = [[piano_cell_instruments(cell) for cell in row] for row in saved_melody]
        else:
            loaded_melody = project.get('melody_pattern')
            if isinstance(loaded_melody, list) and len(loaded_melody) == len(piano_notes) and all((len(row) == num_steps for row in loaded_melody)):
                piano_patterns[0] = [[piano_cell_instruments(cell) for cell in row] for row in loaded_melody]
        piano_scroll = max(0, min(len(piano_notes) - visible_piano_rows, int(project.get('piano_scroll', piano_scroll))))
        loaded_drum_mixer = project.get('drum_mixer')
        if isinstance(loaded_drum_mixer, list) and len(loaded_drum_mixer) == len(drum_tracks):
            for i in range(len(drum_tracks)):
                drum_mixer[i].update(loaded_drum_mixer[i])
        loaded_instrument_mixer = project.get('instrument_mixer')
        if isinstance(loaded_instrument_mixer, dict):
            for name in instruments:
                saved_state = loaded_instrument_mixer.get(name)
                if isinstance(saved_state, dict):
                    instrument_mixer[name].update(saved_state)
        else:
            loaded_melody_mixer = project.get('melody_mixer')
            if isinstance(loaded_melody_mixer, dict):
                for name in instruments:
                    instrument_mixer[name].update(loaded_melody_mixer)
        missing_audio = []
        loaded_audio_bank = project.get('audio_patterns')
        if isinstance(loaded_audio_bank, dict):
            for key, saved_tracks in loaded_audio_bank.items():
                index = parse_pattern_index(key)
                if index is None or not isinstance(saved_tracks, list):
                    continue
                for saved_track in saved_tracks:
                    apply_saved_audio_track(saved_track, audio_patterns[index], missing_audio)
        else:
            for saved_track in project.get('audio_tracks', []):
                apply_saved_audio_track(saved_track, audio_patterns[0], missing_audio)
        loaded_order = project.get('song_order')
        if isinstance(loaded_order, list):
            converted_order = []
            for row in loaded_order:
                if not isinstance(row, dict):
                    continue
                converted_order.append({
                    'drums': parse_pattern_index(row.get('drums')) if row.get('drums') is not None else None,
                    'piano': parse_pattern_index(row.get('piano')) if row.get('piano') is not None else None,
                    'audio': parse_pattern_index(row.get('audio')) if row.get('audio') is not None else None,
                })
            song_order = converted_order
        else:
            has_legacy_audio = bool(project.get('audio_tracks'))
            song_order = [{'drums': 0, 'piano': 0, 'audio': 0 if has_legacy_audio else None}]
        set_edit_drum_pattern(project.get('edit_drum_pattern', 0))
        set_edit_piano_pattern(project.get('edit_piano_pattern', 0))
        set_edit_audio_pattern(project.get('edit_audio_pattern', 0))
        try:
            current_order_index = int(project.get('current_order_index', 0))
        except (TypeError, ValueError):
            current_order_index = 0
        if song_order:
            current_order_index = max(0, min(current_order_index, len(song_order) - 1))
        else:
            current_order_index = 0
        song_scroll = 0
        clamp_song_scroll()
        saved_view = project.get('current_view', 'DRUMS')
        if saved_view == 'SEQUENCER':
            saved_view = 'DRUMS'
        if saved_view == 'ORDER':
            saved_view = 'SONG'
        if saved_view in ('DRUMS', 'PIANO', 'AUDIO', 'SONG', 'MIXER'):
            current_view = saved_view
        dragging_audio = None
        mixer_drag = None
        print('Project loaded:', path)
        if missing_audio:
            print('Missing audio files:')
            for missing_path in missing_audio:
                print(' -', missing_path)
    except Exception as error:
        print("Couldn't load project:", path)
        print(error)


# Microphone recording
def microphone_callback(indata, frames, time_info, status):
    if status:
        print(status)
    if recording_microphone:
        microphone_chunks.append(indata.copy())

def start_microphone_recording():
    global recording_microphone
    global microphone_chunks
    global microphone_stream
    global microphone_record_samplerate
    microphone_chunks = []
    device_index, device_name = get_selected_microphone()
    if device_index is None:
        print('No microphone input device found.')
        return
    try:
        device_info = sd.query_devices(device_index)
        record_samplerate = int(device_info['default_samplerate'])
        record_channels = min(1, device_info['max_input_channels'])
        if record_channels < 1:
            print('Selected device has no input channels:', device_name)
            return
        microphone_record_samplerate = record_samplerate
        microphone_stream = sd.InputStream(device=device_index, samplerate=record_samplerate, channels=record_channels, dtype='float32', callback=microphone_callback)
        microphone_stream.start()
        recording_microphone = True
        print('Recording microphone:', device_name, '@', record_samplerate, 'Hz')
    except Exception as error:
        microphone_stream = None
        recording_microphone = False
        print("Couldn't start microphone:", device_name)
        print(error)

def stop_microphone_recording():
    global recording_microphone
    global microphone_stream
    recording_microphone = False
    if microphone_stream is not None:
        microphone_stream.stop()
        microphone_stream.close()
        microphone_stream = None
    if len(microphone_chunks) == 0:
        print('No microphone audio recorded.')
        return
    recording = np.concatenate(microphone_chunks, axis=0)
    recording = np.clip(recording, -1, 1)
    recording_int16 = (recording * 32767).astype(np.int16)
    filename = time.strftime('JRYBeats_recording_%Y%m%d_%H%M%S.wav')
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(microphone_record_samplerate)
        wav_file.writeframes(recording_int16.tobytes())
    print('Saved recording:', filename)
    import_audio_file(filename)

def choose_audio_file():
    path = ask_open_path('Import Audio', AUDIO_FILETYPES)
    if path:
        import_audio_file(path)

def draw_waveform(waveform, rect, color):
    if len(waveform) == 0:
        return
    center_y = rect.centery
    samples_per_pixel = max(1, len(waveform) // max(1, rect.width))
    for x in range(rect.width):
        start = x * samples_per_pixel
        end = min(start + samples_per_pixel, len(waveform))
        if start >= len(waveform):
            break
        section = waveform[start:end]
        amplitude = np.max(np.abs(section))
        height = int(amplitude * (rect.height / 2 - 4))
        pygame.draw.line(screen, color, (rect.left + x, center_y - height), (rect.left + x, center_y + height), 1)
bpm = 120
min_bpm = 60
max_bpm = 200
playing = False
current_step = 0
next_step_time = 0
current_view = 'DRUMS'
dragging_audio = None

def play_step(step):
    row = active_song_row()
    drums_id = row.get('drums')
    if drums_id is not None:
        drum_pat = drum_patterns[drums_id]
        for track in range(len(drum_tracks)):
            if drum_pat[track][step]:
                play_sound_with_mixer(drum_sounds[track], drum_mixer[track])
    piano_id = row.get('piano')
    if piano_id is not None:
        piano_pat = piano_patterns[piano_id]
        for note_index in range(len(piano_notes)):
            for cell_instrument in piano_cell_instruments(piano_pat[note_index][step]):
                sound = melody_sounds[piano_notes[note_index], cell_instrument]
                play_sound_with_mixer(sound, instrument_mixer[cell_instrument])
    audio_id = row.get('audio')
    if audio_id is not None:
        for audio_track in audio_patterns[audio_id]:
            if audio_track['start_step'] == step:
                audio_track['channel'] = play_sound_with_mixer(audio_track['sound'], audio_track)

def start_playback(now):
    global playing
    global current_step
    global current_order_index
    global next_step_time
    playing = True
    current_step = 0
    current_order_index = 0
    play_step(current_step)
    step_interval = 60000 / bpm / 4
    next_step_time = now + step_interval

def stop_playback():
    global playing
    global current_step
    global current_order_index
    playing = False
    current_step = 0
    current_order_index = 0
    pygame.mixer.stop()
    for track in all_audio_tracks():
        track['channel'] = None

def advance_playback_step():
    global current_step
    global current_order_index
    current_step += 1
    if current_step >= num_steps:
        current_step = 0
        if song_order:
            current_order_index = (current_order_index + 1) % len(song_order)

def handle_song_click(pos):
    global song_order
    global song_scroll
    global current_order_index
    if song_add_rect.collidepoint(pos):
        song_order.append(empty_song_row())
        clamp_song_scroll()
        song_scroll = max(0, len(song_order) - SONG_VISIBLE_ROWS)
        return
    for visible_index in range(SONG_VISIBLE_ROWS):
        order_index = song_scroll + visible_index
        if order_index >= len(song_order):
            break
        rects = song_row_control_rects(visible_index)
        row = song_order[order_index]
        if rects['drums_minus'].collidepoint(pos):
            row['drums'] = cycle_pattern_slot(row.get('drums'), -1)
            return
        if rects['drums_plus'].collidepoint(pos):
            row['drums'] = cycle_pattern_slot(row.get('drums'), 1)
            return
        if rects['piano_minus'].collidepoint(pos):
            row['piano'] = cycle_pattern_slot(row.get('piano'), -1)
            return
        if rects['piano_plus'].collidepoint(pos):
            row['piano'] = cycle_pattern_slot(row.get('piano'), 1)
            return
        if rects['audio_minus'].collidepoint(pos):
            row['audio'] = cycle_pattern_slot(row.get('audio'), -1)
            return
        if rects['audio_plus'].collidepoint(pos):
            row['audio'] = cycle_pattern_slot(row.get('audio'), 1)
            return
        if rects['delete'].collidepoint(pos):
            del song_order[order_index]
            if song_order:
                current_order_index %= len(song_order)
            else:
                current_order_index = 0
            clamp_song_scroll()
            return
play_rect = pygame.Rect(30, 82, 70, 42)
stop_rect = pygame.Rect(115, 82, 70, 42)
bpm_minus_rect = pygame.Rect(215, 82, 35, 42)
bpm_rect = pygame.Rect(250, 82, 70, 42)
bpm_plus_rect = pygame.Rect(320, 82, 35, 42)
drums_tab_rect = pygame.Rect(25, 150, 85, 40)
piano_tab_rect = pygame.Rect(118, 150, 125, 40)
audio_tab_rect = pygame.Rect(251, 150, 80, 40)
song_tab_rect = pygame.Rect(339, 150, 72, 40)
mixer_tab_rect = pygame.Rect(419, 150, 78, 40)
save_project_rect = pygame.Rect(805, 153, 72, 32)
load_project_rect = pygame.Rect(885, 153, 72, 32)
import_audio_rect = pygame.Rect(25, 215, 82, 36)
record_audio_rect = pygame.Rect(115, 215, 88, 36)
drum_pattern_minus_rect, drum_pattern_value_rect, drum_pattern_plus_rect = pattern_selector_rects(58, 204)
piano_pattern_minus_rect, piano_pattern_value_rect, piano_pattern_plus_rect = pattern_selector_rects(850, 199)
audio_pattern_minus_rect, audio_pattern_value_rect, audio_pattern_plus_rect = pattern_selector_rects(790, 215)
song_add_rect = pygame.Rect(30, 605, 90, 28)
mic_prev_rect = pygame.Rect(455, 88, 30, 30)
mic_device_rect = pygame.Rect(490, 88, 420, 30)
mic_next_rect = pygame.Rect(915, 88, 30, 30)

# Main loop
running = True
while running:
    now = pygame.time.get_ticks()
    mouse_position = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.DROPFILE:
            dropped_file = event.file
            import_audio_file(dropped_file)
            current_view = 'AUDIO'
        elif event.type == pygame.MOUSEWHEEL:
            if current_view == 'PIANO':
                piano_scroll -= event.y
                piano_scroll = max(0, min(piano_scroll, len(piano_notes) - visible_piano_rows))
            elif current_view == 'MIXER':
                mixer_scroll_x -= event.x * 50
                mixer_scroll_x -= event.y * 50
                clamp_mixer_scroll()
            elif current_view == 'SONG':
                song_scroll -= event.y
                clamp_song_scroll()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if playing:
                    stop_playback()
                else:
                    start_playback(now)
            elif event.key == pygame.K_LEFT:
                bpm = max(min_bpm, bpm - 5)
            elif event.key == pygame.K_RIGHT:
                bpm = min(max_bpm, bpm + 5)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if play_rect.collidepoint(event.pos):
                if not playing:
                    start_playback(now)
            elif stop_rect.collidepoint(event.pos):
                stop_playback()
            elif bpm_minus_rect.collidepoint(event.pos):
                bpm = max(min_bpm, bpm - 5)
            elif bpm_plus_rect.collidepoint(event.pos):
                bpm = min(max_bpm, bpm + 5)
            elif drums_tab_rect.collidepoint(event.pos):
                current_view = 'DRUMS'
            elif piano_tab_rect.collidepoint(event.pos):
                current_view = 'PIANO'
            elif audio_tab_rect.collidepoint(event.pos):
                current_view = 'AUDIO'
            elif song_tab_rect.collidepoint(event.pos):
                current_view = 'SONG'
            elif mixer_tab_rect.collidepoint(event.pos):
                current_view = 'MIXER'
            elif save_project_rect.collidepoint(event.pos):
                save_project()
            elif load_project_rect.collidepoint(event.pos):
                stop_playback()
                load_project()
            elif current_view == 'AUDIO' and mic_prev_rect.collidepoint(event.pos):
                if microphone_devices and (not recording_microphone):
                    selected_microphone_position = (selected_microphone_position - 1) % len(microphone_devices)
            elif current_view == 'AUDIO' and (mic_next_rect.collidepoint(event.pos) or mic_device_rect.collidepoint(event.pos)):
                if microphone_devices and (not recording_microphone):
                    selected_microphone_position = (selected_microphone_position + 1) % len(microphone_devices)
            elif current_view == 'AUDIO' and import_audio_rect.collidepoint(event.pos):
                choose_audio_file()
            elif current_view == 'AUDIO' and record_audio_rect.collidepoint(event.pos):
                if recording_microphone:
                    stop_microphone_recording()
                else:
                    start_microphone_recording()
            elif current_view == 'MIXER':
                handle_mixer_click(event.pos)
            elif current_view == 'SONG':
                handle_song_click(event.pos)
            elif current_view == 'DRUMS':
                if drum_pattern_minus_rect.collidepoint(event.pos):
                    set_edit_drum_pattern(edit_drum_pattern - 1)
                elif drum_pattern_plus_rect.collidepoint(event.pos):
                    set_edit_drum_pattern(edit_drum_pattern + 1)
                else:
                    for track in range(len(drum_tracks)):
                        y = track_start_y + track * row_height
                        for step in range(num_steps):
                            x = sequencer_start_x + step * (step_size + step_gap)
                            rect = pygame.Rect(x, y + 16, step_size, step_size)
                            if rect.collidepoint(event.pos):
                                pattern[track][step] = not pattern[track][step]
                                if pattern[track][step]:
                                    play_sound_with_mixer(drum_sounds[track], drum_mixer[track])
            elif current_view == 'PIANO':
                if piano_pattern_minus_rect.collidepoint(event.pos):
                    set_edit_piano_pattern(edit_piano_pattern - 1)
                elif piano_pattern_plus_rect.collidepoint(event.pos):
                    set_edit_piano_pattern(edit_piano_pattern + 1)
                else:
                    instrument_y = 585
                    clicked_instrument = False
                    for i in range(len(instruments)):
                        rect = pygame.Rect(130 + i * 100, instrument_y, 85, 30)
                        if rect.collidepoint(event.pos):
                            melody_instrument = instruments[i]
                            clicked_instrument = True
                            break
                    if not clicked_instrument:
                        for visible_row in range(visible_piano_rows):
                            note_index = piano_scroll + visible_row
                            if note_index >= len(piano_notes):
                                continue
                            y = piano_grid_top + visible_row * piano_row_height
                            key_rect = pygame.Rect(20, y, 100, piano_row_height)
                            if key_rect.collidepoint(event.pos):
                                note = piano_notes[note_index]
                                sound = melody_sounds[note, melody_instrument]
                                play_sound_with_mixer(sound, instrument_mixer[melody_instrument])
                            for step in range(num_steps):
                                x = piano_grid_start_x + step * piano_step_width
                                cell_rect = pygame.Rect(x, y, piano_step_width, piano_row_height)
                                if cell_rect.collidepoint(event.pos):
                                    added = toggle_selected_piano_cell(note_index, step)
                                    if added:
                                        note = piano_notes[note_index]
                                        sound = melody_sounds[note, melody_instrument]
                                        play_sound_with_mixer(sound, instrument_mixer[melody_instrument])
            elif current_view == 'AUDIO':
                if audio_pattern_minus_rect.collidepoint(event.pos):
                    set_edit_audio_pattern(edit_audio_pattern - 1)
                elif audio_pattern_plus_rect.collidepoint(event.pos):
                    set_edit_audio_pattern(edit_audio_pattern + 1)
                else:
                    for index in range(len(audio_tracks)):
                        track = audio_tracks[index]
                        y = audio_track_top + index * audio_track_height
                        mute_rect = pygame.Rect(30, y + 20, 30, 25)
                        delete_rect = pygame.Rect(70, y + 20, 30, 25)
                        loop_duration = 60 / bpm / 4 * num_steps
                        clip_width = int(track['length'] / loop_duration * audio_timeline_width)
                        clip_width = max(50, clip_width)
                        clip_width = min(audio_timeline_width, clip_width)
                        clip_x = audio_timeline_x + int(track['start_step'] / num_steps * audio_timeline_width)
                        clip_rect = pygame.Rect(clip_x, y + 10, clip_width, 48)
                        if mute_rect.collidepoint(event.pos):
                            track['muted'] = not track['muted']
                            break
                        elif delete_rect.collidepoint(event.pos):
                            channel = track.get('channel')
                            if channel is not None:
                                channel.stop()
                            del audio_tracks[index]
                            break
                        elif clip_rect.collidepoint(event.pos):
                            dragging_audio = index
                            break
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            dragging_audio = None
            mixer_drag = None
            mixer_scrollbar_drag = None
        elif event.type == pygame.MOUSEMOTION:
            if mixer_scrollbar_drag is not None:
                bar, thumb = mixer_scrollbar_geometry()
                if thumb is not None:
                    travel = bar.width - thumb.width
                    if travel > 0:
                        ratio = (event.pos[0] - mixer_scrollbar_drag - bar.x) / travel
                        mixer_scroll_x = ratio * mixer_max_scroll()
                        clamp_mixer_scroll()
            elif mixer_drag is not None:
                track_state, control_type, control_rect = mixer_drag
                if control_type == 'volume':
                    ratio = (control_rect.bottom - event.pos[1]) / control_rect.height
                    track_state['volume'] = max(0.0, min(1.0, ratio))
                elif control_type == 'pan':
                    ratio = (event.pos[0] - control_rect.left) / control_rect.width
                    track_state['pan'] = max(-1.0, min(1.0, ratio * 2.0 - 1.0))
            elif dragging_audio is not None and dragging_audio < len(audio_tracks):
                relative_x = event.pos[0] - audio_timeline_x
                ratio = relative_x / audio_timeline_width
                ratio = max(0, min(0.999, ratio))
                new_step = int(ratio * num_steps)
                audio_tracks[dragging_audio]['start_step'] = new_step
    if playing:
        step_interval = 60000 / bpm / 4
        while now >= next_step_time:
            advance_playback_step()
            play_step(current_step)
            next_step_time += step_interval
    refresh_playing_audio_mixer()
    clamp_mixer_scroll()
    clamp_song_scroll()
    screen.fill(BACKGROUND)
    title = title_font.render('JRYBeats', True, TEXT_COLOR)
    screen.blit(title, (25, 18))
    pygame.draw.line(screen, LINE_COLOR, (0, 68), (WIDTH, 68), 1)
    pygame.draw.rect(screen, BUTTON_BACKGROUND, play_rect)
    pygame.draw.rect(screen, LINE_COLOR, play_rect, 2)
    pygame.draw.polygon(screen, GREEN, [(53, 91), (53, 115), (79, 103)])
    pygame.draw.rect(screen, BUTTON_BACKGROUND, stop_rect)
    pygame.draw.rect(screen, LINE_COLOR, stop_rect, 2)
    pygame.draw.rect(screen, RED, (140, 92, 20, 20))
    pygame.draw.rect(screen, BUTTON_BACKGROUND, bpm_minus_rect)
    pygame.draw.rect(screen, LINE_COLOR, bpm_minus_rect, 2)
    minus = small_font.render('-', True, TEXT_COLOR)
    screen.blit(minus, minus.get_rect(center=bpm_minus_rect.center))
    pygame.draw.rect(screen, BUTTON_BACKGROUND, bpm_rect)
    pygame.draw.rect(screen, LINE_COLOR, bpm_rect, 2)
    bpm_text = small_font.render(str(bpm), True, TEXT_COLOR)
    screen.blit(bpm_text, bpm_text.get_rect(center=bpm_rect.center))
    pygame.draw.rect(screen, BUTTON_BACKGROUND, bpm_plus_rect)
    pygame.draw.rect(screen, LINE_COLOR, bpm_plus_rect, 2)
    plus = small_font.render('+', True, TEXT_COLOR)
    screen.blit(plus, plus.get_rect(center=bpm_plus_rect.center))
    bpm_label = small_font.render('BPM', True, TEXT_COLOR)
    screen.blit(bpm_label, (365, 93))
    pygame.draw.line(screen, LINE_COLOR, (0, 140), (WIDTH, 140), 1)
    drums_color = TEXT_COLOR if current_view == 'DRUMS' else SECONDARY_TEXT
    piano_color = TEXT_COLOR if current_view == 'PIANO' else SECONDARY_TEXT
    audio_color = TEXT_COLOR if current_view == 'AUDIO' else SECONDARY_TEXT
    song_color = TEXT_COLOR if current_view == 'SONG' else SECONDARY_TEXT
    mixer_color = TEXT_COLOR if current_view == 'MIXER' else SECONDARY_TEXT
    drums_text = section_font.render('DRUMS', True, drums_color)
    piano_text = section_font.render('PIANO ROLL', True, piano_color)
    audio_text = section_font.render('AUDIO', True, audio_color)
    song_text = section_font.render('SONG', True, song_color)
    mixer_text = section_font.render('MIXER', True, mixer_color)
    screen.blit(drums_text, drums_text.get_rect(center=drums_tab_rect.center))
    screen.blit(piano_text, piano_text.get_rect(center=piano_tab_rect.center))
    screen.blit(audio_text, audio_text.get_rect(center=audio_tab_rect.center))
    screen.blit(song_text, song_text.get_rect(center=song_tab_rect.center))
    screen.blit(mixer_text, mixer_text.get_rect(center=mixer_tab_rect.center))
    pygame.draw.rect(screen, BUTTON_BACKGROUND, save_project_rect)
    pygame.draw.rect(screen, LINE_COLOR, save_project_rect, 1)
    pygame.draw.rect(screen, BUTTON_BACKGROUND, load_project_rect)
    pygame.draw.rect(screen, LINE_COLOR, load_project_rect, 1)
    save_text = tiny_font.render('SAVE', True, TEXT_COLOR)
    load_text = tiny_font.render('LOAD', True, TEXT_COLOR)
    screen.blit(save_text, save_text.get_rect(center=save_project_rect.center))
    screen.blit(load_text, load_text.get_rect(center=load_project_rect.center))
    if current_view == 'DRUMS':
        pygame.draw.line(screen, BLUE, (drums_tab_rect.left, 185), (drums_tab_rect.right, 185), 3)
    elif current_view == 'PIANO':
        pygame.draw.line(screen, BLUE, (piano_tab_rect.left, 185), (piano_tab_rect.right, 185), 3)
    elif current_view == 'AUDIO':
        pygame.draw.line(screen, BLUE, (audio_tab_rect.left, 185), (audio_tab_rect.right, 185), 3)
    elif current_view == 'SONG':
        pygame.draw.line(screen, BLUE, (song_tab_rect.left, 185), (song_tab_rect.right, 185), 3)
    else:
        pygame.draw.line(screen, BLUE, (mixer_tab_rect.left, 185), (mixer_tab_rect.right, 185), 3)
    pygame.draw.line(screen, LINE_COLOR, (0, 198), (WIDTH, 198), 1)
    if current_view == 'DRUMS':
        pat_label = tiny_font.render('PAT', True, SECONDARY_TEXT)
        screen.blit(pat_label, (22, 210))
        draw_pattern_selector(drum_pattern_minus_rect.x, drum_pattern_minus_rect.y, edit_drum_pattern)
        for step in range(num_steps):
            x = sequencer_start_x + step * (step_size + step_gap)
            number = step_font.render(str(step + 1), True, TEXT_COLOR)
            screen.blit(number, number.get_rect(center=(x + step_size // 2, 216)))
        if playing:
            playhead_x = sequencer_start_x + current_step * (step_size + step_gap)
            pygame.draw.line(screen, PLAYHEAD_BLUE, (playhead_x + step_size // 2, 225), (playhead_x + step_size // 2, 575), 3)
        for track in range(len(drum_tracks)):
            y = track_start_y + track * row_height
            pygame.draw.line(screen, LINE_COLOR, (20, y + 60), (WIDTH - 20, y + 60), 1)
            screen.blit(drum_images[track], (30, y + 6))
            track_name = track_font.render(drum_tracks[track], True, TEXT_COLOR)
            screen.blit(track_name, (95, y + 20))
            for step in range(num_steps):
                x = sequencer_start_x + step * (step_size + step_gap)
                rect = pygame.Rect(x, y + 16, step_size, step_size)
                if pattern[track][step]:
                    color = GREEN
                    if rect.collidepoint(mouse_position):
                        color = GREEN_HOVER
                else:
                    color = STEP_BACKGROUND
                    if rect.collidepoint(mouse_position):
                        color = STEP_HOVER
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, LINE_COLOR, rect, 2)
    elif current_view == 'PIANO':
        pat_label = tiny_font.render('PAT', True, SECONDARY_TEXT)
        screen.blit(pat_label, (810, 205))
        draw_pattern_selector(piano_pattern_minus_rect.x, piano_pattern_minus_rect.y, edit_piano_pattern)
        for step in range(num_steps):
            x = piano_grid_start_x + step * piano_step_width
            number = step_font.render(str(step + 1), True, TEXT_COLOR)
            screen.blit(number, number.get_rect(center=(x + piano_step_width // 2, 213)))
        for visible_row in range(visible_piano_rows):
            note_index = piano_scroll + visible_row
            if note_index >= len(piano_notes):
                continue
            note = piano_notes[note_index]
            y = piano_grid_top + visible_row * piano_row_height
            sharp = '#' in note
            key_rect = pygame.Rect(20, y, 100, piano_row_height)
            if sharp:
                key_color = BLACK_KEY
                key_text_color = (240, 240, 240)
            else:
                key_color = WHITE_KEY
                key_text_color = TEXT_COLOR
            pygame.draw.rect(screen, key_color, key_rect)
            pygame.draw.rect(screen, LINE_COLOR, key_rect, 1)
            note_text = tiny_font.render(note, True, key_text_color)
            screen.blit(note_text, (35, y + 7))
            for step in range(num_steps):
                x = piano_grid_start_x + step * piano_step_width
                cell_rect = pygame.Rect(x, y, piano_step_width, piano_row_height)
                mouse_over = cell_rect.collidepoint(mouse_position)
                draw_piano_cell(cell_rect, melody_pattern[note_index][step], mouse_over, sharp)
        if playing:
            playhead_x = piano_grid_start_x + current_step * piano_step_width
            pygame.draw.line(screen, PLAYHEAD_BLUE, (playhead_x, piano_grid_top), (playhead_x, piano_grid_top + visible_piano_rows * piano_row_height), 3)
        instrument_y = 585
        for i in range(len(instruments)):
            rect = pygame.Rect(130 + i * 100, instrument_y, 85, 30)
            if melody_instrument == instruments[i]:
                pygame.draw.rect(screen, INSTRUMENT_COLORS[instruments[i]], rect)
                instrument_text_color = (255, 255, 255)
            else:
                pygame.draw.rect(screen, BUTTON_BACKGROUND, rect)
                instrument_text_color = TEXT_COLOR
            pygame.draw.rect(screen, LINE_COLOR, rect, 1)
            label = tiny_font.render(instruments[i], True, instrument_text_color)
            screen.blit(label, label.get_rect(center=rect.center))
    elif current_view == 'SONG':
        header = small_font.render('SONG', True, TEXT_COLOR)
        screen.blit(header, (20, 207))
        if song_order:
            position_text = tiny_font.render(f'{current_order_index + 1:02d}/{len(song_order):02d}', True, SECONDARY_TEXT)
        else:
            position_text = tiny_font.render('--/--', True, SECONDARY_TEXT)
        screen.blit(position_text, (170, 212))
        pos_header = tiny_font.render('POS', True, SECONDARY_TEXT)
        drums_header = tiny_font.render('DRUMS', True, SECONDARY_TEXT)
        piano_header = tiny_font.render('PIANO', True, SECONDARY_TEXT)
        audio_header = tiny_font.render('AUDIO', True, SECONDARY_TEXT)
        screen.blit(pos_header, (30, 230))
        screen.blit(drums_header, (150, 230))
        screen.blit(piano_header, (350, 230))
        screen.blit(audio_header, (550, 230))
        if not song_order:
            empty_text = small_font.render('Empty song. Add a row or playback uses the current patterns.', True, SECONDARY_TEXT)
            screen.blit(empty_text, (30, 280))
        for visible_index in range(SONG_VISIBLE_ROWS):
            order_index = song_scroll + visible_index
            if order_index >= len(song_order):
                break
            rects = song_row_control_rects(visible_index)
            y = rects['y']
            row = song_order[order_index]
            if playing and order_index == current_order_index:
                pygame.draw.rect(screen, (230, 238, 250), pygame.Rect(20, y - 4, 760, SONG_ROW_HEIGHT - 2))
            pos_text = tiny_font.render(f'{order_index + 1:02d}', True, TEXT_COLOR)
            screen.blit(pos_text, (30, y + 6))
            draw_pattern_selector(118, y, row.get('drums'), allow_empty=True)
            draw_pattern_selector(318, y, row.get('piano'), allow_empty=True)
            draw_pattern_selector(518, y, row.get('audio'), allow_empty=True)
            pygame.draw.rect(screen, BUTTON_BACKGROUND, rects['delete'])
            pygame.draw.rect(screen, LINE_COLOR, rects['delete'], 1)
            delete_text = tiny_font.render('X', True, RED)
            screen.blit(delete_text, delete_text.get_rect(center=rects['delete'].center))
        pygame.draw.rect(screen, BUTTON_BACKGROUND, song_add_rect)
        pygame.draw.rect(screen, LINE_COLOR, song_add_rect, 1)
        add_text = tiny_font.render('+ ADD', True, TEXT_COLOR)
        screen.blit(add_text, add_text.get_rect(center=song_add_rect.center))
    elif current_view == 'MIXER':
        mixer_tracks = get_mixer_tracks()
        header = small_font.render('TRACK MIXER', True, TEXT_COLOR)
        screen.blit(header, (20, 207))
        pygame.draw.rect(screen, BUTTON_BACKGROUND, mixer_scroll_left_rect)
        pygame.draw.rect(screen, LINE_COLOR, mixer_scroll_left_rect, 1)
        pygame.draw.rect(screen, BUTTON_BACKGROUND, mixer_scroll_right_rect)
        pygame.draw.rect(screen, LINE_COLOR, mixer_scroll_right_rect, 1)
        left_arrow = tiny_font.render('<', True, TEXT_COLOR)
        right_arrow = tiny_font.render('>', True, TEXT_COLOR)
        screen.blit(left_arrow, left_arrow.get_rect(center=mixer_scroll_left_rect.center))
        screen.blit(right_arrow, right_arrow.get_rect(center=mixer_scroll_right_rect.center))
        for i, (name, track_state) in enumerate(mixer_tracks):
            x = mixer_strip_screen_x(i)
            if x + MIXER_STRIP_INNER_WIDTH < MIXER_STRIP_START_X or x > MIXER_VIEW_RIGHT:
                continue
            strip_rect = pygame.Rect(x, MIXER_STRIP_TOP, MIXER_STRIP_INNER_WIDTH, MIXER_STRIP_HEIGHT)
            pygame.draw.rect(screen, BUTTON_BACKGROUND, strip_rect)
            pygame.draw.rect(screen, LIGHT_LINE, strip_rect, 1)
            if name in INSTRUMENT_COLORS:
                pygame.draw.rect(screen, INSTRUMENT_COLORS[name], pygame.Rect(x, MIXER_STRIP_TOP, MIXER_STRIP_INNER_WIDTH, 4))
                name_color = INSTRUMENT_COLORS[name]
            else:
                name_color = TEXT_COLOR
            name_text = tiny_font.render(name[:11], True, name_color)
            screen.blit(name_text, name_text.get_rect(center=(x + 48, MIXER_STRIP_TOP + 15)))
            mute_rect, solo_rect, volume_rect, pan_rect = mixer_control_rects(x)
            mute_color = RED if track_state.get('muted', False) else BUTTON_BACKGROUND
            solo_color = GREEN if track_state.get('solo', False) else BUTTON_BACKGROUND
            pygame.draw.rect(screen, mute_color, mute_rect)
            pygame.draw.rect(screen, LINE_COLOR, mute_rect, 1)
            pygame.draw.rect(screen, solo_color, solo_rect)
            pygame.draw.rect(screen, LINE_COLOR, solo_rect, 1)
            mute_text = tiny_font.render('M', True, TEXT_COLOR)
            solo_text = tiny_font.render('S', True, TEXT_COLOR)
            screen.blit(mute_text, mute_text.get_rect(center=mute_rect.center))
            screen.blit(solo_text, solo_text.get_rect(center=solo_rect.center))
            pygame.draw.rect(screen, LIGHT_LINE, volume_rect)
            volume = track_state.get('volume', 1.0)
            knob_y = int(volume_rect.bottom - volume * volume_rect.height)
            pygame.draw.rect(screen, BLUE, pygame.Rect(x + 37, knob_y - 5, 30, 10))
            volume_text = tiny_font.render(str(int(volume * 100)), True, SECONDARY_TEXT)
            screen.blit(volume_text, volume_text.get_rect(center=(x + 52, MIXER_STRIP_TOP + 325)))
            pygame.draw.line(screen, LIGHT_LINE, (pan_rect.left, pan_rect.centery), (pan_rect.right, pan_rect.centery), 3)
            pan = track_state.get('pan', 0.0)
            pan_x = int(pan_rect.left + (pan + 1.0) / 2.0 * pan_rect.width)
            pygame.draw.circle(screen, PURPLE, (pan_x, pan_rect.centery), 7)
            pan_label = tiny_font.render('PAN', True, SECONDARY_TEXT)
            screen.blit(pan_label, pan_label.get_rect(center=(x + 50, MIXER_STRIP_TOP + 378)))
        bar, thumb = mixer_scrollbar_geometry()
        pygame.draw.rect(screen, LIGHT_LINE, bar, border_radius=4)
        if thumb is not None:
            pygame.draw.rect(screen, BLUE, thumb, border_radius=4)
        else:
            pygame.draw.rect(screen, (220, 220, 215), bar, border_radius=4)
    elif current_view == 'AUDIO':
        pygame.draw.rect(screen, BUTTON_BACKGROUND, mic_prev_rect)
        pygame.draw.rect(screen, LINE_COLOR, mic_prev_rect, 1)
        pygame.draw.rect(screen, BUTTON_BACKGROUND, mic_device_rect)
        pygame.draw.rect(screen, LINE_COLOR, mic_device_rect, 1)
        pygame.draw.rect(screen, BUTTON_BACKGROUND, mic_next_rect)
        pygame.draw.rect(screen, LINE_COLOR, mic_next_rect, 1)
        previous_mic_text = small_font.render('<', True, TEXT_COLOR)
        next_mic_text = small_font.render('>', True, TEXT_COLOR)
        screen.blit(previous_mic_text, previous_mic_text.get_rect(center=mic_prev_rect.center))
        screen.blit(next_mic_text, next_mic_text.get_rect(center=mic_next_rect.center))
        selected_device_index, selected_device_name = get_selected_microphone()
        display_mic_name = selected_device_name
        while tiny_font.size('MIC: ' + display_mic_name)[0] > mic_device_rect.width - 18 and len(display_mic_name) > 4:
            display_mic_name = display_mic_name[:-1]
        if display_mic_name != selected_device_name:
            display_mic_name = display_mic_name[:-3] + '...'
        microphone_selector_text = tiny_font.render('MIC: ' + display_mic_name, True, TEXT_COLOR)
        screen.blit(microphone_selector_text, microphone_selector_text.get_rect(center=mic_device_rect.center))
        pygame.draw.rect(screen, BUTTON_BACKGROUND, import_audio_rect)
        pygame.draw.rect(screen, LINE_COLOR, import_audio_rect, 2)
        import_text = tiny_font.render('IMPORT', True, TEXT_COLOR)
        screen.blit(import_text, import_text.get_rect(center=import_audio_rect.center))
        record_button_color = BUTTON_BACKGROUND
        if recording_microphone:
            record_button_color = RED
        pygame.draw.rect(screen, record_button_color, record_audio_rect)
        pygame.draw.rect(screen, LINE_COLOR, record_audio_rect, 2)
        if recording_microphone:
            record_label = 'STOP'
        else:
            record_label = 'MIC'
        record_text = tiny_font.render(record_label, True, TEXT_COLOR)
        icon_x = record_audio_rect.x + 8
        icon_y = record_audio_rect.centery - microphone_image.get_height() // 2
        screen.blit(microphone_image, (icon_x, icon_y))
        record_text_rect = record_text.get_rect(midleft=(icon_x + microphone_image.get_width() + 5, record_audio_rect.centery))
        screen.blit(record_text, record_text_rect)
        pat_label = tiny_font.render('PAT', True, SECONDARY_TEXT)
        screen.blit(pat_label, (750, 221))
        draw_pattern_selector(audio_pattern_minus_rect.x, audio_pattern_minus_rect.y, edit_audio_pattern)
        for step in range(num_steps):
            x = audio_timeline_x + int(step / num_steps * audio_timeline_width)
            pygame.draw.line(screen, LIGHT_LINE, (x, 255), (x, HEIGHT - 25), 1)
            number = tiny_font.render(str(step + 1), True, SECONDARY_TEXT)
            screen.blit(number, (x + 4, 238))
        if playing:
            playhead_x = audio_timeline_x + int(current_step / num_steps * audio_timeline_width)
            pygame.draw.line(screen, PLAYHEAD_BLUE, (playhead_x, 255), (playhead_x, HEIGHT - 20), 3)
        for index in range(len(audio_tracks)):
            track = audio_tracks[index]
            y = audio_track_top + index * audio_track_height
            pygame.draw.line(screen, LIGHT_LINE, (20, y + 62), (WIDTH - 20, y + 62), 1)
            mute_rect = pygame.Rect(30, y + 20, 30, 25)
            mute_color = RED if track['muted'] else BUTTON_BACKGROUND
            pygame.draw.rect(screen, mute_color, mute_rect)
            pygame.draw.rect(screen, LINE_COLOR, mute_rect, 1)
            mute_text = tiny_font.render('M', True, TEXT_COLOR)
            screen.blit(mute_text, mute_text.get_rect(center=mute_rect.center))
            delete_rect = pygame.Rect(70, y + 20, 30, 25)
            pygame.draw.rect(screen, BUTTON_BACKGROUND, delete_rect)
            pygame.draw.rect(screen, LINE_COLOR, delete_rect, 1)
            delete_text = tiny_font.render('X', True, RED)
            screen.blit(delete_text, delete_text.get_rect(center=delete_rect.center))
            name_text = tiny_font.render(track['name'][:16], True, TEXT_COLOR)
            screen.blit(name_text, (108, y + 25))
            loop_duration = 60 / bpm / 4 * num_steps
            clip_width = int(track['length'] / loop_duration * audio_timeline_width)
            clip_width = max(50, clip_width)
            clip_width = min(audio_timeline_width, clip_width)
            clip_x = audio_timeline_x + int(track['start_step'] / num_steps * audio_timeline_width)
            clip_rect = pygame.Rect(clip_x, y + 10, clip_width, 48)
            pygame.draw.rect(screen, (240, 220, 195), clip_rect)
            pygame.draw.rect(screen, ORANGE, clip_rect, 2)
            draw_waveform(track['waveform'], clip_rect.inflate(-6, -6), ORANGE)
    pygame.display.update()
    clock.tick(60)
pygame.quit()
