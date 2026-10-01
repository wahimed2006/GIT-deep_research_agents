"""Voice request handler for the deep research agent.

This module provides the Ina class, which serves as the top-level entry point
for voice-driven research. It captures voice input from the user, transcribes it
locally using faster-whisper, and orchestrates the full research pipeline.
"""

from __future__ import annotations

import logging
import queue
import sys
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
from scipy.signal import resample_poly
import sounddevice as sd
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
"""Voice detection threshold (amplitude). Lower values = more sensitive."""

DEFAULT_SILENCE_DURATION: float = 1.2
"""Duration of silence (seconds) before triggering transcription."""

DEFAULT_MAX_RECORDING_SEC: float = 20.0
"""Maximum recording duration (seconds) to prevent infinite capture."""

WHISPER_MODEL_SIZE: str = "small"
"""Whisper model size ('tiny', 'base', 'small', 'medium', 'large')."""

WHISPER_DEVICE: str = "cpu"
"""Device for Whisper inference ('cpu', 'cuda', 'auto')."""

WHISPER_COMPUTE_TYPE: str = "int8"
"""Computation precision ('int8', 'float16', 'float32')."""

EXIT_KEYWORDS: set[str] = {
    "quit", "quitter", "exit", "stop", "arreter", "arrêter", "au revoir", "bye"
}
"""Voice keywords that trigger graceful shutdown."""

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# INA CLASS
# =============================================================================

class Ina:
    """Voice request handler and primary entry point for deep research.

    Ina manages the entire voice-driven research cycle:
        1. Captures microphone stream via sounddevice.
        2. Transcribes voice locally using faster-whisper.
        3. Forwards the transcribed query to the ResearchOrchestrator.
        4. Prints the generated synthesis and returns to listening.

    Attributes:
        orchestrator: The underlying ResearchOrchestrator instance.
        model: WhisperModel instance for local transcription.
        device_index: Sounddevice index of the target microphone.
        energy_threshold: Minimum amplitude considered as active speech.
        silence_duration: Seconds of silence after speech to trigger transcription.
        max_recording_sec: Timeout ceiling for a single utterance.
        channels: Resolved hardware channel count (1 for mono, 2 for stereo).
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        device_index: int = DEFAULT_CAPTURE_DEVICE,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
        silence_duration: float = DEFAULT_SILENCE_DURATION,
        max_recording_sec: float = DEFAULT_MAX_RECORDING_SEC,
        on_transcription: Optional[Callable[[str], None]] = None,
    ):
        """Initialize Ina and resolve dependencies.

        Args:
            orchestrator: Optional existing ResearchOrchestrator instance.
                          If None, it is automatically imported and initialized.
            device_index: Sounddevice index for the capture microphone.
            model_size: Whisper model size. Defaults to 'small'.
            device: Computation device ('cpu', 'cuda', 'auto').
            compute_type: Precision type ('int8', 'float16', 'float32').
            energy_threshold: Amplitude threshold for Voice Activity Detection.
            silence_duration: Seconds of trailing silence before inference.
            max_recording_sec: Maximum recording duration per capture.
            on_transcription: Optional callback invoked with the transcribed text.
        """
        print(f"[Ina] Initializing Whisper model ({model_size}, {compute_type})...")
        self.device_index = device_index
        self.capture_sample_rate = self._resolve_device_sample_rate()
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.max_recording_sec = max_recording_sec
        self.on_transcription = on_transcription

        self.audio_queue: queue.Queue = queue.Queue()
        self.is_listening: bool = False

        # 1. Initialize Whisper Model
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=4,
        )

        # 2. Resolve Microphone Channels (Prevents invalid channel count errors on USB mics)
        self.channels = self._resolve_device_channels()

        # 3. Provision Orchestrator if not provided
        if orchestrator is None:
            self.orchestrator = self._init_default_orchestrator()
        else:
            self.orchestrator = orchestrator

        print(
            f"[Ina] System ready. Device: {self.device_index} ({self.channels} ch, "
            f"{self.capture_sample_rate:g} Hz), VAD Threshold: {self.energy_threshold}"
        )

    def _resolve_device_sample_rate(self) -> float:
        """Return the selected device's native input sample rate."""
        try:
            device_info = sd.query_devices(self.device_index, "input")
            return float(device_info.get("default_samplerate", CAPTURE_SAMPLE_RATE))
        except Exception as e:
            logger.warning(
                f"Could not query sample rate for device {self.device_index}: {e}. "
                f"Using {CAPTURE_SAMPLE_RATE} Hz."
            )
            return float(CAPTURE_SAMPLE_RATE)

    def _resolve_device_channels(self) -> int:
        """Resolve maximum input channels supported by the selected device."""
        try:
            device_info = sd.query_devices(self.device_index, "input")
            max_ch = int(device_info.get("max_input_channels", 1))
            # Most USB microphones operate in 1 or 2 channels natively
            return min(max_ch, 2) if max_ch > 0 else 1
        except Exception as e:
            logger.warning(f"Could not query device {self.device_index}: {e}. Defaulting to 1 channel.")
            return 1

    def _init_default_orchestrator(self) -> Any:
        """Create the repository's complete multi-agent research pipeline."""
        try:
            if __package__:
                from ..agents.orchestrator import ResearchOrchestrator
            else:
                project_root = str(Path(__file__).resolve().parents[2])
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                from src.agents.orchestrator import ResearchOrchestrator
        except ImportError as err:
            logger.exception("Could not import the research orchestrator")
            print(f"[Ina] Orchestrator import failed: {err}")
            return None

        try:
            orchestrator = ResearchOrchestrator()
            print("[Ina] Orchestrator ready: decomposition, search, analysis, scraping and synthesis.")
            return orchestrator
        except Exception as err:
            logger.exception("Could not initialize the research orchestrator")
            print(
                "[Ina] Orchestrator initialization failed. "
                f"Check Ollama and its models ({err})."
            )
            return None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Sounddevice stream callback for buffering audio blocks."""
        if status:
            logger.warning(f"Audio stream status: {status}")

        # Downmix stereo to mono on-the-fly if needed
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = indata.mean(axis=1)
        else:
            mono = indata[:, 0]

        self.audio_queue.put(mono.copy())

    def _listen_once(self) -> str:
        """Capture microphone input until silence is detected, then transcribe."""
        while not self.audio_queue.empty():
            self.audio_queue.get()

        recorded_blocks: list[np.ndarray] = []
        has_spoken: bool = False
        silence_blocks_count: int = 0

        block_size = max(1, round(self.capture_sample_rate * 0.1))
        block_duration_sec: float = block_size / self.capture_sample_rate
        max_silence_blocks: int = int(self.silence_duration / block_duration_sec)
        max_recording_blocks: int = int(self.max_recording_sec / block_duration_sec)

        print("\n[Ina] Écoute active... Posez votre question.")

        with sd.InputStream(
            samplerate=self.capture_sample_rate,
            blocksize=block_size,
            channels=self.channels,
            dtype="float32",
            device=self.device_index,
            callback=self._audio_callback,
        ):
            for _ in range(max_recording_blocks):
                block = self.audio_queue.get()
                amplitude = float(np.abs(block).mean())

                if amplitude > self.energy_threshold:
                    if not has_spoken:
                        print("[Ina] Voix détectée...")
                        has_spoken = True
                    silence_blocks_count = 0
                    recorded_blocks.append(block)
                elif has_spoken:
                    recorded_blocks.append(block)
                    silence_blocks_count += 1

                    if silence_blocks_count >= max_silence_blocks:
                        break

        if not recorded_blocks:
            return ""

        audio_data: np.ndarray = np.concatenate(recorded_blocks, axis=0)

        # Resample from capture rate (44.1kHz) to Whisper rate (16kHz)
        audio_data = resample_poly(
            audio_data,
            SAMPLE_RATE,
            round(self.capture_sample_rate),
        ).astype(np.float32)

        # Normalize gain to prevent low-energy transcription artifacts
        max_peak: float = float(np.max(np.abs(audio_data)))
        if max_peak > 0.005:
            audio_data = audio_data / max_peak

        # Direct in-memory transcription
        segments, _ = self.model.transcribe(
            audio_data,
            language="fr",
            beam_size=3,
            vad_filter=True,
        )

        return " ".join(seg.text for seg in segments).strip()

    def listen_once(self) -> str:
        """Capture a single vocal phrase with error boundary handling."""
        self.is_listening = True
        try:
            return self._listen_once()
        except Exception as e:
            logger.error(f"Voice capture error: {e}")
            print(f"[Ina] Capture error: {e}")
            return ""
        finally:
            self.is_listening = False

    def execute_query(self, query: str) -> Optional[Any]:
        """Send the transcribed query through the research orchestrator."""
        if not self.orchestrator:
            print(f"[Ina] No orchestrator loaded. Query: '{query}'")
            return None

        print(f"\n[Ina] Envoi à l'orchestrateur : \"{query}\"")
        try:
            if hasattr(self.orchestrator, "research_and_print"):
                return self.orchestrator.research_and_print(query)
            if hasattr(self.orchestrator, "research"):
                result = self.orchestrator.research(query)
                print(f"\n[Ina] Réponse :\n{result.answer}")
                return result
            elif hasattr(self.orchestrator, "process_request"):
                return self.orchestrator.process_request(query)
            elif hasattr(self.orchestrator, "run"):
                return self.orchestrator.run(query)
            elif callable(self.orchestrator):
                return self.orchestrator(query)
            else:
                print("[Ina] Error: Configured orchestrator provides no recognized execution method.")
        except Exception as e:
            logger.error(f"Orchestrator execution error: {e}")
            print(f"[Ina] Pipeline failed: {e}")
        return None

    def run(self) -> None:
        """Start the persistent voice-driven interaction loop.

        This method blocks and continuously listens for user queries, triggers
        the orchestrator workflow upon speech completion, and exits cleanly
        when a termination keyword ('quitter', 'exit', etc.) or Ctrl+C is received.
        """
        print("\n" + "=" * 60)
        print("INA - Deep Research Agent (Voice Interface)")
        print("Dites votre question ou 'quitter' pour arrêter (Ctrl+C pour sortir)")
        print("=" * 60)

        try:
            while True:
                query = self.listen_once()

                if not query:
                    continue

                print(f"[Vous] : {query}")

                # Check for termination keywords
                cleaned = query.lower().strip(" .!?,;")
                if cleaned in EXIT_KEYWORDS:
                    print("\n[Ina] Arrêt demandé. Au revoir !")
                    break

                if self.on_transcription:
                    self.on_transcription(query)

                # Execute the full multi-agent research pipeline
                self.execute_query(query)

        except KeyboardInterrupt:
            print("\n[Ina] Session interrompue par l'utilisateur.")
        except Exception as e:
            logger.error(f"Fatal error in Ina.run loop: {e}")
            raise


if __name__ == "__main__":
    Ina().run()