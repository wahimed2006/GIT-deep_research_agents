"""Local ultra-fast speech recognition module using faster-whisper and sounddevice.

This module provides the VoiceToTextModel class for real-time voice capture and
transcription using the faster-whisper library. It features automatic silence
detection, multi-channel audio support, and sample rate conversion for USB
microphones.

Features:
    - Real-time audio capture via sounddevice with callback
    - Automatic silence detection with configurable threshold
    - Resampling from 44.1kHz (USB mic) to 16kHz (Whisper requirement)
    - Audio normalization for consistent transcription quality
    - Voice Activity Detection (VAD) filtering

Example:
    >>> # Initialize and use
    >>> voice_model = VoiceToTextModel(device_index=3)
    >>> text = voice_model.listen_and_transcribe()
    >>> print(f"You said: {text}")
    
    >>> # Custom configuration
    >>> voice_model = VoiceToTextModel(
    ...     device_index=3,
    ...     energy_threshold=0.005,
    ...     silence_duration=1.5
    ... )
    >>> text = voice_model.listen_and_transcribe()
"""

from __future__ import annotations

import queue
import logging
from typing import Optional
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from faster_whisper import WhisperModel


# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE: int = 16000
"""Target sample rate for Whisper transcription (16kHz)."""

DEFAULT_CAPTURE_DEVICE: int = 3
"""Default capture device index (fifine USB microphone)."""

CAPTURE_SAMPLE_RATE: int = 44100
"""Native sample rate of the USB microphone (44.1kHz)."""

BLOCK_SIZE: int = 4410
"""Audio block size in samples (100ms at 44.1kHz)."""

DEFAULT_ENERGY_THRESHOLD: float = 0.001
"""Voice detection threshold (amplitude).
Lower values = more sensitive. Typical range: 0.001-0.01.
"""

DEFAULT_SILENCE_DURATION: float = 1.2
"""Duration of silence (seconds) before triggering transcription."""

DEFAULT_MAX_RECORDING_SEC: float = 20
"""Maximum recording duration (seconds) to prevent infinite capture."""

WHISPER_MODEL_SIZE: str = "small"
"""Whisper model size ('tiny', 'base', 'small', 'medium', 'large')."""

WHISPER_DEVICE: str = "cpu"
"""Device for Whisper inference ('cpu', 'cuda', 'auto')."""

WHISPER_COMPUTE_TYPE: str = "int8"
"""Computation precision ('int8', 'float16', 'float32')."""


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# VOICETOTEXTMODEL CLASS
# =============================================================================

class VoiceToTextModel:
    """Real-time voice capture and transcription using faster-whisper.
    
    This class provides real-time voice capture and transcription using the
    faster-whisper library. It features automatic silence detection, multi-channel
    audio support, and sample rate conversion for USB microphones.
    
    Features:
        - Real-time audio capture via sounddevice with callback
        - Automatic silence detection with configurable threshold
        - Resampling from 44.1kHz (USB mic) to 16kHz (Whisper requirement)
        - Audio normalization for consistent transcription quality
        - Voice Activity Detection (VAD) filtering
        - Multi-device support (USB microphones, built-in mics)
    
    Attributes:
        model: WhisperModel instance for transcription.
        device_index: Sounddevice device index for the microphone.
        audio_queue: Thread-safe queue for audio blocks.
        energy_threshold: Voice detection threshold (amplitude).
        silence_duration: Duration of silence before transcription.
        max_recording_sec: Maximum recording duration.
    
    Example:
        >>> # Basic usage
        >>> voice_model = VoiceToTextModel(device_index=3)
        >>> text = voice_model.listen_and_transcribe()
        >>> if text:
        ...     print(f"You said: {text}")
        
        >>> # Custom configuration
        >>> voice_model = VoiceToTextModel(
        ...     device_index=3,
        ...     energy_threshold=0.005,
        ...     silence_duration=1.5,
        ...     model_size="base"
        ... )
        >>> text = voice_model.listen_and_transcribe()
        
        >>> # List available devices
        >>> voice_model = VoiceToTextModel()
        >>> voice_model.list_audio_devices()
    """

    def __init__(
        self,
        device_index: int = DEFAULT_CAPTURE_DEVICE,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
        silence_duration: float = DEFAULT_SILENCE_DURATION,
        max_recording_sec: float = DEFAULT_MAX_RECORDING_SEC,
    ):
        """Initialize VoiceToTextModel.
        
        Args:
            device_index: Sounddevice device index for the microphone.
                         Use sd.query_devices() to find available devices.
                         Default: 3 (fifine USB microphone).
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
                       'small' provides good accuracy/speed balance.
                       'base' is faster but less accurate.
            device: Device for Whisper inference ('cpu', 'cuda', 'auto').
            compute_type: Computation precision ('int8', 'float16', 'float32').
            energy_threshold: Voice detection threshold (amplitude).
                            Lower = more sensitive. Typical: 0.001-0.01.
            silence_duration: Duration of silence (seconds) before transcription.
            max_recording_sec: Maximum recording duration (seconds).
        
        Example:
            >>> # Default configuration
            >>> voice_model = VoiceToTextModel()
            >>> 
            >>> # Custom device and threshold
            >>> voice_model = VoiceToTextModel(
            ...     device_index=0,
            ...     energy_threshold=0.005,
            ...     silence_duration=1.5
            ... )
            >>> 
            >>> # Faster model (base instead of small)
            >>> voice_model = VoiceToTextModel(
            ...     model_size="base",
            ...     compute_type="int8"
            ... )
        """
        print(f"[VoiceToText] Loading Whisper model ({model_size}, {compute_type})...")
        
        self.device_index = device_index
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.max_recording_sec = max_recording_sec
        
        # Initialize audio queue
        self.audio_queue: queue.Queue = queue.Queue()
        
        # Initialize Whisper model
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=4,
        )
        
        # Recording state
        self.is_listening: bool = False
        
        print(f"[VoiceToText] Ready. Device: {device_index}, Threshold: {energy_threshold}")

    def _audio_callback(
        self,
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
        # Convert stereo to mono if needed, or extract mono channel
        mono = indata.mean(axis=1) if indata.ndim > 1 and indata.shape[1] > 1 else indata[:, 0]
        self.audio_queue.put(mono.copy())

    def _listen_and_transcribe(self) -> str:
        """Internal method: Listen until end of phrase and transcribe.
        
        This method captures audio from the microphone until silence is detected
        after speech, then transcribes using Whisper. It handles:
        - Voice activity detection with configurable threshold
        - Automatic silence detection
        - Sample rate conversion (44.1kHz -> 16kHz)
        - Audio normalization
        
        Returns:
            Transcribed text as a string.
            Empty string if no speech was detected.
        
        Note:
            This method blocks until:
            - Silence is detected after speech (transcription triggered)
            - max_recording_sec seconds elapse (timeout)
            - No speech is detected (returns empty string)
        """
        # Clear any pending audio in queue
        while not self.audio_queue.empty():
            self.audio_queue.get()

        recorded_blocks: list[np.ndarray] = []
        has_spoken: bool = False
        silence_blocks_count: int = 0
        
        # Calculate block counts from durations
        block_duration_sec: float = BLOCK_SIZE / CAPTURE_SAMPLE_RATE
        max_silence_blocks: int = int(self.silence_duration / block_duration_sec)
        max_recording_blocks: int = int(self.max_recording_sec / block_duration_sec)

        print("\n[Micro] Listening... Speak now.")

        # Open audio input stream with callback
        with sd.InputStream(
            samplerate=CAPTURE_SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            device=self.device_index,
            callback=self._audio_callback,
        ):
            for _ in range(max_recording_blocks):
                block = self.audio_queue.get()
                amplitude = float(np.abs(block).mean())

                # Detect voice activity
                if amplitude > self.energy_threshold:
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
                        print(f"[Micro] Silence detected ({self.silence_duration}s)")
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
            logger.debug(f"Audio normalized (peak: {max_peak:.6f} -> 1.0)")
            print(f"[Audio] Normalized (peak: {max_peak:.6f} -> 1.0)")

        # Transcribe directly from numpy array (no WAV file needed)
        segments, info = self.model.transcribe(
            audio_data,
            language="fr",
            beam_size=3,
            vad_filter=True,
        )

        full_text: str = " ".join(seg.text for seg in segments).strip()
        logger.info(f"Transcription done (language: {info.language}, length: {len(full_text)})")
        print(f"[Transcription] Done (language: {info.language})")
        
        return full_text

    def listen_and_transcribe(self) -> str:
        """Listen for one voice command and return transcribed text.
        
        This is the public method for capturing a single voice command.
        It wraps _listen_and_transcribe() with error handling.
        
        Returns:
            Transcribed text as a string.
            Empty string if no speech was detected or an error occurred.
        
        Example:
            >>> voice_model = VoiceToTextModel(device_index=3)
            >>> text = voice_model.listen_and_transcribe()
            >>> if text:
            ...     print(f"You said: {text}")
        """
        self.is_listening = True
        try:
            return self._listen_and_transcribe()
        except Exception as e:
            logger.error(f"Voice capture error: {e}")
            print(f"[Error] Voice capture failed: {e}")
            return ""
        finally:
            self.is_listening = False

    def listen_loop(self, max_iterations: Optional[int] = None) -> None:
        """Continuously listen for voice commands.
        
        This method runs an infinite loop (or limited iterations) capturing
        voice commands and printing the transcribed text.
        
        Args:
            max_iterations: Maximum number of iterations (None = infinite).
                           Use Ctrl+C to stop the loop.
        
        Example:
            >>> voice_model = VoiceToTextModel(device_index=3)
            >>> voice_model.listen_loop()  # Infinite loop
            >>> # Or limited:
            >>> voice_model.listen_loop(max_iterations=10)
        """
        print(f"\n[VoiceToText] Starting listen loop (max: {max_iterations or 'infinite'} iterations)")
        print("[VoiceToText] Press Ctrl+C to stop.\n")
        
        iteration = 0
        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1
                logger.debug(f"Listen iteration {iteration}")
                
                text = self.listen_and_transcribe()
                
                if text:
                    print(f"\n[Transcribed text]: {text}\n")
                else:
                    print("\n[Info] Nothing captured.\n")
                
        except KeyboardInterrupt:
            print("\n[VoiceToText] Listen loop stopped by user.")
        except Exception as e:
            logger.error(f"Listen loop error: {e}")
            raise

    def list_audio_devices(self) -> list[dict]:
        """List available audio input devices.
        
        This method queries sounddevice for all audio input devices
        and returns their information.
        
        Returns:
            List of dicts with device information:
            - index: Device index for device_index parameter
            - name: Device name
            - channels: Number of input channels
            - samplerate: Default sample rate
        
        Example:
            >>> voice_model = VoiceToTextModel()
            >>> devices = voice_model.list_audio_devices()
            >>> for dev in devices:
            ...     print(f"{dev['index']}: {dev['name']} ({dev['channels']} ch)")
        """
        devices = sd.query_devices()
        input_devices = [
            {
                "index": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "samplerate": d["default_samplerate"],
            }
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
        
        print("\n=== Available Audio Input Devices ===\n")
        for dev in input_devices:
            print(f"  {dev['index']}: {dev['name']} ({dev['channels']} ch, {dev['samplerate']} Hz)")
        print()
        
        return input_devices


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=== VoiceToTextModel Test ===\n")
    
    # Initialize VoiceToTextModel
    voice_model = VoiceToTextModel(
        device_index=DEFAULT_CAPTURE_DEVICE,  # Adjust to your device
        model_size="small",
        compute_type="int8",
        energy_threshold=0.001,
    )
    
    # List available devices
    voice_model.list_audio_devices()
    
    print("Press Enter to start listening...")
    input()
    
    # Single listen test
    print("\n[Test] Listening for one command...")
    text = voice_model.listen_and_transcribe()
    
    if text:
        print(f"\n[Success] Transcribed: {text}")
    else:
        print("\n[Info] No speech detected.")