"""Local ultra-fast speech recognition module using faster-whisper and sounddevice.

This module provides real-time voice capture and transcription using the 
faster-whisper library. It features automatic silence detection, multi-channel
audio support, and sample rate conversion for USB microphones.

Features:
    - Real-time audio capture via sounddevice with callback
    - Automatic silence detection with configurable threshold
    - Resampling from 44.1kHz (USB mic) to 16kHz (Whisper requirement)
    - Audio normalization for consistent transcription quality
    - Voice Activity Detection (VAD) filtering

Example:
    >>> from faster_whisper import WhisperModel
    >>> model = WhisperModel("small", device="cpu", compute_type="int8")
    >>> text = listen_and_transcribe(model, device_index=2)
    >>> print(f"You said: {text}")
"""

import queue
import sys
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from typing import Callable


# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE: int = 16000
"""Target sample rate for Whisper transcription (16kHz)."""

CAPTURE_DEVICE: int = 2
"""Default capture device index (fifine USB microphone).
Use sd.query_devices() to find your device index.
"""

CAPTURE_SAMPLE_RATE: int = 44100
"""Native sample rate of the USB microphone (44.1kHz)."""

BLOCK_SIZE: int = 4410
"""Audio block size in samples (100ms at 44.1kHz)."""

ENERGY_THRESHOLD: float = 0.001
"""Voice detection threshold (amplitude).
Lower values = more sensitive. Typical range: 0.001-0.01.
"""

SILENCE_DURATION_SEC: float = 1.2
"""Duration of silence (seconds) before triggering transcription."""

MAX_RECORDING_SEC: float = 20
"""Maximum recording duration (seconds) to prevent infinite capture."""


# =============================================================================
# GLOBAL STATE
# =============================================================================

audio_queue: queue.Queue = queue.Queue()
"""Thread-safe queue for audio blocks from the callback."""


# =============================================================================
# AUDIO CAPTURE
# =============================================================================

def audio_callback(
    indata: np.ndarray,
    frames: int,
    time_info: dict,
    status: sd.CallbackFlags,
) -> None:
    """Capture audio blocks from the microphone continuously.
    
    This callback function is called by sounddevice for each audio block.
    It converts stereo input to mono and pushes to the audio queue.
    
    Args:
        indata: Audio buffer of shape (frames, channels).
        frames: Number of audio frames in this block.
        time_info: Timing information (ignored).
        status: Callback status flags (warnings/errors).
    
    Note:
        This function runs in a separate thread and must be fast.
        It only copies audio data to the queue, no processing.
    """
    if status:
        print(f"[Audio Warning] {status}", file=sys.stderr)
    
    # Convert stereo to mono if needed, or extract mono channel
    mono = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
    audio_queue.put(mono.copy())


# =============================================================================
# SPEECH DETECTION AND TRANSCRIPTION
# =============================================================================

def listen_and_transcribe(
    model: WhisperModel,
    device_index: int = CAPTURE_DEVICE,
) -> str:
    """Listen until end of phrase and transcribe the audio signal.
    
    This function captures audio from the microphone until silence is detected
    after speech, then transcribes using Whisper. It handles:
    - Voice activity detection with configurable threshold
    - Automatic silence detection
    - Sample rate conversion (44.1kHz -> 16kHz)
    - Audio normalization
    
    Args:
        model: Pre-loaded WhisperModel instance for transcription.
        device_index: Sounddevice device index for the microphone.
                     Use sd.query_devices() to find available devices.
    
    Returns:
        Transcribed text as a string.
        Empty string if no speech was detected.
    
    Example:
        >>> model = WhisperModel("small", device="cpu", compute_type="int8")
        >>> text = listen_and_transcribe(model, device_index=2)
        >>> if text:
        ...     print(f"You said: {text}")
    
    Note:
        The function blocks until:
        - Silence is detected after speech (transcription triggered)
        - MAX_RECORDING_SEC seconds elapse (timeout)
        - No speech is detected (returns empty string)
    """
    # Clear any pending audio in queue
    while not audio_queue.empty():
        audio_queue.get()

    recorded_blocks: list[np.ndarray] = []
    has_spoken: bool = False
    silence_blocks_count: int = 0
    
    # Calculate block counts from durations
    max_silence_blocks: int = int(
        SILENCE_DURATION_SEC / (BLOCK_SIZE / CAPTURE_SAMPLE_RATE)
    )
    max_recording_blocks: int = int(
        MAX_RECORDING_SEC / (BLOCK_SIZE / CAPTURE_SAMPLE_RATE)
    )

    print("\n[Micro] Listening... Speak now.")

    # Open audio input stream with callback
    with sd.InputStream(
        samplerate=CAPTURE_SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        device=device_index,
        callback=audio_callback,
    ):
        for _ in range(max_recording_blocks):
            block = audio_queue.get()
            amplitude = float(np.abs(block).mean())

            # Detect voice activity
            if amplitude > ENERGY_THRESHOLD:
                if not has_spoken:
                    print("[Micro] Voice detected...")
                    has_spoken = True
                silence_blocks_count = 0
                recorded_blocks.append(block)
            elif has_spoken:
                # Continue recording during silence after speech
                recorded_blocks.append(block)
                silence_blocks_count += 1
                
                # Stop if silence duration reached
                if silence_blocks_count >= max_silence_blocks:
                    print(f"[Micro] Silence detected ({SILENCE_DURATION_SEC}s)")
                    break

    # Return empty if no speech detected
    if not recorded_blocks:
        print("[Micro] No speech detected.")
        return ""

    # Assemble audio buffer from blocks
    audio_data: np.ndarray = np.concatenate(recorded_blocks, axis=0)

    # Resample from capture rate (44.1kHz) to Whisper rate (16kHz)
    audio_data = resample_poly(
        audio_data, 
        SAMPLE_RATE, 
        CAPTURE_SAMPLE_RATE
    ).astype(np.float32)

    # Normalize audio to prevent quiet voice issues
    max_peak: float = np.max(np.abs(audio_data))
    if max_peak > 0.005:
        audio_data = audio_data / max_peak
        print(f"[Audio] Normalized (peak: {max_peak:.6f} -> 1.0)")

    # Transcribe directly from numpy array (no WAV file needed)
    segments, info = model.transcribe(
        audio_data,
        language="fr",
        beam_size=3,
        vad_filter=True,
    )

    full_text: str = " ".join(seg.text for seg in segments).strip()
    print(f"[Transcription] Done (language: {info.language})")
    
    return full_text


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=== VoiceRecognizer (sounddevice) ===\n")
    
    print("Loading Whisper model (small, int8)...")
    whisper_model = WhisperModel(
        model_size_or_path="small",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
    )

    print("Ready. Speak into the microphone (Ctrl+C to quit).\n")
    
    try:
        while True:
            text = listen_and_transcribe(whisper_model, device_index=CAPTURE_DEVICE)
            if text:
                print(f"\n[Transcribed text]: {text}\n")
            else:
                print("\n[Info] Nothing captured.\n")
    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")