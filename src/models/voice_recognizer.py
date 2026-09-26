"""Local ultra-fast speech recognition module using faster-whisper.

This module provides a VoiceRecognizer class for capturing audio from a microphone
and transcribing it to text using the faster-whisper library. It supports:
- Automatic silence detection for hands-free recording
- Multiple Whisper model sizes for speed/accuracy trade-offs
- Real-time audio amplitude monitoring
- Device selection for multi-microphone setups
- Automatic mono conversion for multi-channel devices

Example:
    >>> recognizer = VoiceRecognizer(model_size="base", language="fr")
    >>> text = recognizer.listen()
    >>> print(f"You said: {text}")
"""

from __future__ import annotations

import io
import wave
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from typing import Optional, Tuple


class VoiceRecognizer:
    """Audio capture and local transcription manager.
    
    This class handles microphone input, silence detection, and speech-to-text
    transcription using the faster-whisper library. It automatically stops
    recording when silence is detected after speech begins.
    
    Features:
        - Automatic silence detection with configurable threshold and duration
        - Support for multiple Whisper model sizes (tiny to large)
        - Multi-device support with automatic mono conversion
        - Real-time amplitude monitoring for debugging
        - Voice Activity Detection (VAD) filtering
        - Configurable beam search for better accuracy
    
    Attributes:
        model: WhisperModel instance for transcription.
        language: Language code for transcription (e.g., 'fr', 'en').
        sample_rate: Audio sample rate in Hz (default: 16000).
        input_device: Device index for microphone selection (None = default).
        silence_threshold: Amplitude threshold for silence detection.
        silence_duration: Duration of silence to trigger stop (seconds).
        max_duration: Maximum recording duration (seconds).
    
    Example:
        >>> # Initialize with default device
        >>> recognizer = VoiceRecognizer(
        ...     model_size="base",
        ...     language="en"
        ... )
        >>> 
        >>> # Initialize with specific microphone (device 0)
        >>> recognizer = VoiceRecognizer(
        ...     model_size="base",
        ...     language="fr",
        ...     input_device=0
        ... )
        >>> text = recognizer.listen_and_print()
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "fr",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = 16000,
        input_device: Optional[int] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        """Initialize Whisper model and audio parameters.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
                       'base' : ultra-fast (~200ms on Ryzen Zen 4), accurate for French.
                       'small' : slightly heavier, even more accurate.
                       'medium' : high accuracy, slower (~1-2s).
                       'large' : maximum accuracy, slowest (~3-5s).
            language: Language code for transcription (e.g., 'fr', 'en', 'es', 'de').
            device: Device type for inference ('cpu', 'cuda', 'auto').
                   'cpu' : CPU-only inference (compatible, slower).
                   'cuda' : GPU inference with CUDA (requires NVIDIA GPU).
                   'auto' : Automatically select best available device.
            compute_type: Computation precision ('int8', 'float16', 'float32').
                         'int8' : Fastest, lowest memory, slight accuracy loss.
                         'float16' : Balanced speed/accuracy (GPU recommended).
                         'float32' : Maximum accuracy, slower, more memory.
            sample_rate: Audio sample rate in Hz (default: 16000).
                        Must be 16000 for Whisper models.
            input_device: Device index for microphone selection (default: None = system default).
                         Use print_audio_devices() to list available devices.
                         Recommended: start with None (default), then try specific devices.
            beam_size: Beam search size for transcription (default: 5).
                      Higher = better accuracy, slower. Range: 1-10.
                      Recommended: 5 for base/small, 10 for medium/large.
            vad_filter: Enable Voice Activity Detection (default: True).
                      Filters out non-speech segments for cleaner transcription.
        
        Example:
            >>> # Use CPU with base model and default microphone
            >>> recognizer = VoiceRecognizer(
            ...     model_size="base",
            ...     language="fr",
            ...     device="cpu",
            ...     compute_type="int8"
            ... )
            >>> 
            >>> # Use specific microphone (device 0) with small model
            >>> recognizer = VoiceRecognizer(
            ...     model_size="small",
            ...     language="en",
            ...     input_device=0,
            ...     beam_size=5
            ... )
        """
        print(f"[Audio] Loading Whisper ({model_size}, {compute_type})...")
        self.language = language
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=4
        )
        
        # Audio recording parameters
        self.chunk_duration = 0.1  # 100ms blocks
        self.silence_threshold = 0.01
        self.silence_duration = 1.2
        self.max_duration = 20.0

    def record_until_silence(
        self,
        silence_threshold: Optional[float] = None,
        silence_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ) -> np.ndarray:
        """Record from microphone until prolonged silence is detected.
        
        This method captures audio in 100ms chunks and monitors amplitude to detect
        speech. Recording starts when amplitude exceeds the threshold and stops after
        the specified duration of silence. The method automatically handles multi-channel
        devices by converting to mono.
        
        Args:
            silence_threshold: Amplitude threshold for silence detection (0.0-1.0).
                              Lower values = more sensitive (default: 0.01).
                              Typical speech: 0.1-0.9, silence: <0.01.
                              If no speech detected, try lowering to 0.005.
            silence_duration: Duration of silence to trigger stop in seconds (default: 1.2).
                             Longer = more tolerant of pauses, shorter = faster stop.
            max_duration: Maximum recording duration in seconds (default: 20.0).
                        Prevents infinite recording if silence is never detected.
        
        Returns:
            Numpy array of audio samples (float32, mono, 16kHz).
            Shape: (n_samples,) for mono audio.
            Empty array (size=0) if no speech detected or error occurred.
        
        Raises:
            sd.PortAudioError: If microphone is unavailable or access denied.
        
        Note:
            This method uses automatic channel detection (channels=None) to avoid
            compatibility issues with devices that report incorrect channel counts.
            Multi-channel input is automatically converted to mono using averaging.
        
        Example:
            >>> # Record with default settings
            >>> audio = recognizer.record_until_silence()
            >>> 
            >>> # Record with custom sensitivity
            >>> audio = recognizer.record_until_silence(
            ...     silence_threshold=0.005,  # More sensitive
            ...     silence_duration=2.0,     # Wait 2s of silence
            ...     max_duration=30.0         # Max 30s recording
            ... )
            >>> if audio.size > 0:
            ...     text = recognizer.transcribe(audio)
        """
        threshold = silence_threshold or self.silence_threshold
        duration = silence_duration or self.silence_duration
        max_dur = max_duration or self.max_duration
        
        print("\n[Micro] Speak now (auto-stop on silence)...")

        chunk_size = int(self.sample_rate * self.chunk_duration)
        silence_chunks_needed = int(duration / self.chunk_duration)

        recorded_chunks = []
        silence_chunk_count = 0
        has_started_speaking = False
        total_chunks = int(max_dur / self.chunk_duration)

        try:
            # Use channels=None for auto-detection (avoids PaErrorCode -9998)
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=None,  # Auto-detect channels from device
                dtype="float32",
                device=self.input_device
            ) as stream:
                for i in range(total_chunks):
                    chunk, _ = stream.read(chunk_size)
                    
                    # Convert to mono if stereo/multi-channel
                    if chunk.ndim > 1 and chunk.shape[1] > 1:
                        chunk = chunk.mean(axis=1, keepdims=True)
                    
                    amplitude = float(np.abs(chunk).mean())

                    if amplitude > threshold:
                        has_started_speaking = True
                        silence_chunk_count = 0
                    elif has_started_speaking:
                        silence_chunk_count += 1

                    recorded_chunks.append(chunk)

                    if has_started_speaking and silence_chunk_count >= silence_chunks_needed:
                        break

        except sd.PortAudioError as e:
            print(f"[Error] Microphone unavailable: {e}")
            print("[Debug] Try using input_device=None (default) or input_device=2 (pipewire)")
            return np.array([], dtype="float32")

        if not has_started_speaking:
            print("[Micro] No sound detected.")
            return np.array([], dtype="float32")

        # Convert to flat mono array
        audio = np.concatenate(recorded_chunks, axis=0)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        
        print(f"[Debug] Recorded {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")
        return audio

    def transcribe(
        self,
        audio_data: np.ndarray,
        print_progress: bool = False,
    ) -> str:
        """Transcribe raw audio data to text using Whisper.
        
        This method converts numpy audio array to text using the faster-whisper model.
        It supports voice activity detection (VAD) to filter out non-speech segments
        and improve transcription quality.
        
        Args:
            audio_data: Numpy array of audio samples (float32, mono, 16kHz).
                       Must be recorded at 16kHz sample rate for Whisper compatibility.
                       Shape should be (n_samples,) for mono audio.
            print_progress: Whether to print transcription progress and debug info
                           (default: False).
        
        Returns:
            Transcribed text as a string, or empty string if audio is empty.
            Text is stripped of leading/trailing whitespace.
            Returns empty string if audio_data.size == 0.
        
        Note:
            The method passes numpy arrays directly to faster-whisper, which handles
            the conversion internally. No manual WAV encoding is required.
        
        Example:
            >>> # Record and transcribe
            >>> audio = recognizer.record_until_silence()
            >>> if audio.size > 0:
            ...     text = recognizer.transcribe(audio, print_progress=True)
            ...     print(f"Transcribed: {text}")
            >>> 
            >>> # Quick transcription without progress
            >>> text = recognizer.transcribe(audio)
        """
        if audio_data.size == 0:
            return ""

        if print_progress:
            print("[Transcription] In progress...")

        segments, info = self.model.transcribe(
            audio_data,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )

        full_text = " ".join(seg.text for seg in segments).strip()
        
        if print_progress:
            print(f"[Transcription] Done (detected language: {info.language})")
            print(f"[Debug] Text length: {len(full_text)} chars")

        return full_text

    def listen(self) -> str:
        """Direct method: listen to mic and return transcribed text.
        
        This is a convenience method that combines record_until_silence() and
        transcribe() into a single call. It automatically prints transcription
        progress for user feedback.
        
        Returns:
            Transcribed text as a string, or empty string if no speech detected.
            Empty string is returned if:
            - No speech was detected during recording
            - Microphone was unavailable
            - Transcription produced no text
        
        Example:
            >>> # Simple listen and get text
            >>> text = recognizer.listen()
            >>> if text:
            ...     print(f"You said: {text}")
            >>> else:
            ...     print("No speech detected")
        """
        audio = self.record_until_silence()
        if audio.size == 0:
            return ""
        return self.transcribe(audio, print_progress=True)
    
    def listen_and_print(self) -> str:
        """Listen and display transcribed text with user feedback.
        
        This method provides a complete user experience with prompts and
        formatted output showing the recognized text. It's designed for
        interactive use and testing.
        
        Returns:
            Transcribed text as a string, or empty string if no speech detected.
            The text is also printed to stdout with a checkmark (✓) or cross (✗).
        
        Example:
            >>> # Interactive dictation
            >>> text = recognizer.listen_and_print()
            # Output:
            # [Dictation mode] Speak freely, stops on silence...
            # [Micro] Speak now (auto-stop on silence)...
            # [Transcription] In progress...
            # ✓ You said: Bonjour, comment ça va?
        """
        print("\n[Dictation mode] Speak freely, stops on silence...\n")
        text = self.listen()
        
        if text:
            print(f"\n✓ You said: {text}\n")
        else:
            print("\n✗ No text recognized\n")
        
        return text
    
    def get_audio_devices(self) -> Tuple[list, list]:
        """List available audio devices on the system.
        
        This method queries sounddevice for all audio input and output devices,
        filtering to only return devices with available channels. Use this to
        discover available microphones and speakers.
        
        Returns:
            Tuple containing two lists:
            - input_devices: List of dicts with microphone device info.
                            Each dict contains: name, index, max_input_channels, etc.
            - output_devices: List of dicts with speaker device info.
                             Each dict contains: name, index, max_output_channels, etc.
        
        Example:
            >>> input_devs, output_devs = recognizer.get_audio_devices()
            >>> print("Microphones:")
            >>> for dev in input_devs:
            ...     print(f"  - {dev['name']} ({dev['max_input_channels']} channels)")
        """
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        return input_devices, output_devices
    
    def print_audio_devices(self) -> None:
        """Display available audio devices in a formatted list.
        
        This method prints all input (microphone) and output (speaker) devices
        with their indices and channel counts. Useful for selecting the correct
        input_device index for initialization.
        
        The output format is designed for easy reading and device selection.
        Device indices can be passed to VoiceRecognizer(input_device=N).
        
        Example:
            >>> recognizer.print_audio_devices()
            # Output:
            # === Audio Devices ===
            # 
            # Input devices (microphones):
            #   0. fifine Microphone: USB Audio (1 channels)
            #   1. HD-Audio Generic: ALC294 Analog (2 channels)
            #   2. pipewire (64 channels)
            #   3. default (64 channels)
            # 
            # Output devices (speakers):
            #   0. HD-Audio Generic: HDMI 0 (8 channels)
            #   1. HD-Audio Generic: ALC294 Analog (2 channels)
        """
        input_devices, output_devices = self.get_audio_devices()
        
        print("\n=== Audio Devices ===\n")
        
        if input_devices:
            print("Input devices (microphones):")
            for i, device in enumerate(input_devices):
                channels = device.get('max_input_channels', 0)
                print(f"  {i}. {device['name']} ({channels} channels)")
        else:
            print("  No input devices found.")
        
        if output_devices:
            print("\nOutput devices (speakers):")
            for i, device in enumerate(output_devices):
                channels = device.get('max_output_channels', 0)
                print(f"  {i}. {device['name']} ({channels} channels)")
        else:
            print("\n  No output devices found.")
        
        print()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=== VoiceRecognizer Test ===\n")
    
    # Initialize with small model for better accuracy
    recognizer = VoiceRecognizer(
        model_size="small",  # Changed from "base" to "small"
        language="fr",
        device="cpu",
        compute_type="int8",
        input_device=None,  # Use system default
        beam_size=5,  # Better beam search
        vad_filter=True,
    )
    
    # List available devices
    recognizer.print_audio_devices()
    
    # Wait for user to start
    print("Press Enter to start recording...")
    input()
    
    # Recording instructions
    print("\n💡 Tips for better transcription:")
    print("   - Speak clearly and at normal pace")
    print("   - Minimize background noise")
    print("   - Keep microphone 10-20cm from mouth")
    print("   - Pause briefly between sentences\n")
    
    # Record and transcribe
    text = recognizer.listen_and_print()
    
    # Display result
    if text:
        print(f"Transcribed text: {text}")