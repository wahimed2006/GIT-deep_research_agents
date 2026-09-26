import queue
import sys
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CAPTURE_DEVICE = 2         # Micro USB fifine (voir sd.query_devices())
CAPTURE_SAMPLE_RATE = 44100
BLOCK_SIZE = 4410           # Blocs de 100 ms a la frequence native du micro
ENERGY_THRESHOLD = 0.001    # Seuil de declenchement de la voix (ajuster si besoin)
SILENCE_DURATION_SEC = 1.2 # Duree de silence avant de lancer la transcription
MAX_RECORDING_SEC = 20      # Evite de rester bloque si la voix n'est jamais detectee

audio_queue: queue.Queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    """Capture les blocs audio du micro en continu."""
    if status:
        print(f"[Audio Warning] {status}", file=sys.stderr)
    # Moyenne des canaux si entree stereo -> mono
    mono = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
    audio_queue.put(mono.copy())


def listen_and_transcribe(model: WhisperModel, device_index=CAPTURE_DEVICE) -> str:
    """Ecoute jusqu'a la fin d'une phrase et transcrit le signal."""
    # Vider la file d'attente
    while not audio_queue.empty():
        audio_queue.get()

    recorded_blocks = []
    has_spoken = False
    silence_blocks_count = 0
    max_silence_blocks = int(
        SILENCE_DURATION_SEC / (BLOCK_SIZE / CAPTURE_SAMPLE_RATE)
    )
    max_recording_blocks = int(MAX_RECORDING_SEC / (BLOCK_SIZE / CAPTURE_SAMPLE_RATE))

    print("\n[Micro] En ecoute... Parlez maintenant.")

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

            if amplitude > ENERGY_THRESHOLD:
                if not has_spoken:
                    print("[Micro] Voix detectee...")
                    has_spoken = True
                silence_blocks_count = 0
                recorded_blocks.append(block)
            elif has_spoken:
                recorded_blocks.append(block)
                silence_blocks_count += 1
                if silence_blocks_count >= max_silence_blocks:
                    # L'utilisateur a fini sa phrase
                    break

    if not recorded_blocks:
        return ""

    # Assemblage du buffer audio
    audio_data = np.concatenate(recorded_blocks, axis=0)

    # Le micro USB capture en 44.1 kHz, tandis que Whisper attend du 16 kHz.
    audio_data = resample_poly(audio_data, SAMPLE_RATE, CAPTURE_SAMPLE_RATE).astype(
        np.float32
    )

    # Normalisation du signal pour eviter les voix trop faibles
    max_peak = np.max(np.abs(audio_data))
    if max_peak > 0.005:
        audio_data = audio_data / max_peak

    # Transcription directe du buffer NumPy (sans passer par un fichier WAV)
    segments, _ = model.transcribe(
        audio_data,
        language="fr",
        beam_size=3,
        vad_filter=True,
    )

    return " ".join(seg.text for seg in segments).strip()


if __name__ == "__main__":
    print("Initialisation du modele Whisper...")
    whisper_model = WhisperModel(
        model_size_or_path="small",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
    )

    print("Pret. Parlez dans le micro (Ctrl+C pour quitter).")
    try:
        while True:
            text = listen_and_transcribe(whisper_model)
            if text:
                print(f"[Texte transcrit] : {text}")
            else:
                print("[Info] Rien n'a ete capture.")
    except KeyboardInterrupt:
        print("\nArret du programme.")