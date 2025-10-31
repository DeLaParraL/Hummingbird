#!/usr/bin/env python3
"""
Hummingbird 
---------------
This script listens to your hum and turns it into structured MIDI notes.

I built this as part of learning how to bridge machine learning, audio processing,
and creativity. The main idea is: record yourself humming a melody,
and Hummingbird figures out the notes, key, and scale — then snaps everything
to the right pitches and exports a MIDI file you can open in Logic, GarageBand, etc.
"""

import argparse
import numpy as np
import librosa
import soundfile as sf
import pretty_midi as pm

# ------------------------------------------------------------
# STEP 1: Some helper data structures
# ------------------------------------------------------------

# I created a simple NoteEvent class to hold MIDI-style info.
class NoteEvent:
    def __init__(self, pitch, start, end, velocity=80):
        self.pitch = pitch        # MIDI pitch number (e.g. 60 = middle C)
        self.start = start        # start time in seconds
        self.end = end            # end time in seconds
        self.velocity = velocity  # how hard the note is played (default 80)

# ------------------------------------------------------------
# STEP 2: Estimate pitch (f0) from the audio
# ------------------------------------------------------------

def estimate_f0(y, sr):
    """
    Uses librosa's pyin (pitch detection algorithm) to find the fundamental frequency (f0)
    across frames. This returns pitch estimates in Hz.

    I'm giving it a range from C2 (~65 Hz) to C7 (~2 kHz) to catch most human humming voices.
    """
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    # pyin can output NaNs where it's unvoiced (not humming)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256
    )
    hop_duration = 256 / sr
    return f0, hop_duration


# ------------------------------------------------------------
# STEP 3: Convert frequency → MIDI pitch numbers
# ------------------------------------------------------------

def hz_to_midi_series(f0):
    """
    Converts f0 in Hz to MIDI note numbers (float),
    e.g. A4 = 440 Hz → 69.
    """
    midi = np.full_like(f0, np.nan)
    valid = ~np.isnan(f0)
    midi[valid] = librosa.hz_to_midi(f0[valid])
    return midi


# ------------------------------------------------------------
# STEP 4: Turn the continuous pitch curve into discrete notes
# ------------------------------------------------------------

def extract_notes(midi_series, hop, min_length=0.08):
    """
    Turns a smooth MIDI curve into note events.
    The logic is: when pitch stays roughly the same → one note.
    When it changes → start a new note.
    """

    notes = []
    current_pitch = None
    start_time = 0

    for i, pitch in enumerate(midi_series):
        t = i * hop
        if np.isnan(pitch):
            # End the note when there's a gap (silence)
            if current_pitch is not None:
                end_time = t
                if end_time - start_time >= min_length:
                    notes.append(NoteEvent(int(round(current_pitch)), start_time, end_time))
                current_pitch = None
            continue

        if current_pitch is None:
            # Start a new note
            current_pitch = pitch
            start_time = t
        else:
            # If the pitch drifts too far, that’s a new note
            if abs(pitch - current_pitch) > 0.6:
                end_time = t
                if end_time - start_time >= min_length:
                    notes.append(NoteEvent(int(round(current_pitch)), start_time, end_time))
                current_pitch = pitch
                start_time = t

    # Catch the last note at the end
    if current_pitch is not None:
        notes.append(NoteEvent(int(round(current_pitch)), start_time, len(midi_series) * hop))
    return notes


# ------------------------------------------------------------
# STEP 5: Detect key + scale (so I can snap notes in tune)
# ------------------------------------------------------------

def detect_key(notes):
    """
    A very rough key detector:
    it counts how often each pitch class (C, C#, D, ...) appears,
    then compares to "templates" for major/minor scales.
    """
    pcs = [n.pitch % 12 for n in notes]
    hist = np.bincount(pcs, minlength=12)
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    best_score = -np.inf
    best = ("C", "major")

    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    for tonic in range(12):
        maj_score = np.corrcoef(hist, np.roll(major_profile, -tonic))[0, 1]
        min_score = np.corrcoef(hist, np.roll(minor_profile, -tonic))[0, 1]
        if maj_score > best_score:
            best_score = maj_score
            best = (note_names[tonic], "major")
        if min_score > best_score:
            best_score = min_score
            best = (note_names[tonic], "minor")

    return best


def scale_pcs(tonic, mode):
    """
    Returns the allowed pitch classes for a given key.
    I use this to snap out-of-scale notes back into the key.
    """
    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    t = note_names.index(tonic)
    if mode == "major":
        degrees = [0,2,4,5,7,9,11]
    else:
        degrees = [0,2,3,5,7,8,10]
    return [(t + d) % 12 for d in degrees]


def snap_to_scale(midi_pitch, allowed):
    """
    If a note isn’t in key, move it to the closest allowed note.
    """
    base = midi_pitch % 12
    if base in allowed:
        return midi_pitch
    for d in range(1, 12):
        if (base + d) % 12 in allowed:
            return midi_pitch + d
        if (base - d) % 12 in allowed:
            return midi_pitch - d
    return midi_pitch


# ------------------------------------------------------------
# STEP 6: Quantize timing (optional)
# ------------------------------------------------------------

def quantize(notes, bpm, grid=16):
    """
    Rounds note starts/ends to the nearest grid division (e.g. 1/16 notes).
    This helps clean up human timing before exporting to MIDI.
    """
    spb = 60 / bpm  # seconds per beat
    grid_sec = spb / (grid / 4)
    qnotes = []
    for n in notes:
        s = round(n.start / grid_sec) * grid_sec
        e = round(n.end / grid_sec) * grid_sec
        if e <= s:
            e = s + grid_sec
        qnotes.append(NoteEvent(n.pitch, s, e, n.velocity))
    return qnotes


# ------------------------------------------------------------
# STEP 7: Export everything as a MIDI file
# ------------------------------------------------------------

def export_midi(notes, bpm, path):
    """
    Creates a PrettyMIDI object and writes it to disk.
    I picked program=0 (Acoustic Grand Piano), but you can change it
    to other GM instruments later.
    """
    midi = pm.PrettyMIDI(initial_tempo=bpm)
    inst = pm.Instrument(program=0)
    for n in notes:
        inst.notes.append(pm.Note(velocity=n.velocity, pitch=n.pitch,
                                  start=n.start, end=n.end))
    midi.instruments.append(inst)
    midi.write(path)
    print(f"✅ Wrote MIDI to {path}")


# ------------------------------------------------------------
# STEP 8: Full pipeline
# ------------------------------------------------------------

def process(audio_path, output_path):
    """
    Full end-to-end conversion:
    1. Load audio
    2. Estimate pitch
    3. Convert to MIDI notes
    4. Detect key + snap to scale
    5. Quantize and export to MIDI
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    print(f"Loaded {audio_path} (sample rate: {sr})")

    f0, hop = estimate_f0(y, sr)
    midi_series = hz_to_midi_series(f0)
    notes = extract_notes(midi_series, hop)
    print(f"Detected {len(notes)} notes before snapping.")

    tonic, mode = detect_key(notes)
    allowed = scale_pcs(tonic, mode)
    snapped = [NoteEvent(snap_to_scale(n.pitch, allowed), n.start, n.end) for n in notes]

    print(f"Detected key: {tonic} {mode}")

    bpm = float(np.median(librosa.beat.tempo(y=y, sr=sr))) if len(y) > 0 else 120.0
    quantized = quantize(snapped, bpm)

    export_midi(quantized, bpm, output_path)


# ------------------------------------------------------------
# STEP 9: Command line entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hummingbird: hum → MIDI converter")
    parser.add_argument("audio", help="Path to your humming audio file (wav, aiff, etc.)")
    parser.add_argument("--out", default="hummingbird.mid", help="Output MIDI filename")
    args = parser.parse_args()

    process(args.audio, args.out)
