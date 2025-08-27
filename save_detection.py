import os
import csv
import warnings

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def lire_detections_csv(csv_path):
    """
    Lit un CSV de la forme :
    index, t_start, t_end, f_start_hz, f_end_hz
    (avec ou sans en-tête)
    Retourne une liste de dicts : start (s), end (s), f_min (kHz), f_max (kHz).
    """
    detections = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row:
                continue
            # Gère un éventuel en-tête non numérique
            if first:
                first = False
                try:
                    float(row[1])
                except ValueError:
                    # ligne d'en-tête -> on la saute
                    continue

            # index, t_start, t_end, f_start_hz, f_end_hz
            _, t_start, t_end, f_start_hz, f_end_hz = row[:5]
            t_start = float(t_start)
            t_end   = float(t_end)
            f_min_khz = float(f_start_hz) / 1000.0
            f_max_khz = float(f_end_hz)   / 1000.0

            detections.append({
                "start": t_start,
                "end": t_end,
                "f_min": f_min_khz,  # en kHz
                "f_max": f_max_khz,  # en kHz
            })
    return detections


def save_one_png_per_detection(audio_file, detections, out_dir="detections",
                               pad_t=0.3, pad_f=5.0, n_fft=4096, hop_length=512,
                               target_sr=None):
    """
    - pad_t (s) : marge de temps autour de la détection (extraction du segment audio).
    - pad_f (kHz) : marge de fréquence autour du rectangle (affichage).
    - target_sr : None = garde le sr d'origine (recommandé si > 80 kHz d'intérêt).
                  ⚠ Doit être >= 2 * max(f_max) en Hz pour ne pas perdre les hautes fréquences.
    - out_dir : dossier de sortie pour les PNG.
    """
    os.makedirs(out_dir, exist_ok=True)
    SCALE = 1000.0  # kHz -> Hz pour l'axe Y

    # Vérif Nyquist si target_sr est imposé
    if detections:
        max_f_khz = max(det["f_max"] for det in detections)
        required_sr = 2 * max_f_khz * SCALE
        if target_sr is not None and target_sr < required_sr:
            warnings.warn(
                f"target_sr={target_sr} Hz < 2×{max_f_khz:.2f} kHz = {required_sr:.0f} Hz. "
                "Les hautes fréquences seront perdues."
            )

    for i, det in enumerate(detections, start=1):
        print(det)
        # Fenêtre temporelle locale
        offset = max(det["start"] - pad_t, 0.0)
        duration = (det["end"] - det["start"]) + 2 * pad_t

        # Charge uniquement ce segment (mono = éco RAM)
        y, sr = librosa.load(audio_file, sr=target_sr, mono=True,
                             offset=offset, duration=duration)

        # Spectrogramme du segment
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        # Affichage
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                                 x_axis='time', y_axis='hz')
        plt.colorbar(format="%+2.0f dB")

        # Coords temps locales
        x0 = det["start"] - offset
        w  = det["end"]   - det["start"]

        # Conversion kHz -> Hz pour l'axe Y
        y0_hz = det["f_min"] * SCALE
        h_hz  = (det["f_max"] - det["f_min"]) * SCALE
        pad_f_hz = pad_f * SCALE

        # Rectangle et zoom
        ax = plt.gca()
        ax.add_patch(Rectangle((x0, y0_hz), w, h_hz,
                               linewidth=2, edgecolor="red", facecolor="none"))

        plt.xlim(max(0, x0 - pad_t/2), x0 + w + pad_t/2)
        plt.ylim(max(0, y0_hz - pad_f_hz), y0_hz + h_hz + pad_f_hz)

        # Fichier de sortie
        out_name = os.path.join(
            out_dir,
            f"detection_{i:02d}_{det['start']:.3f}-{det['end']:.3f}s.png"
        )
        plt.title(
            f"Détection {i}  ({det['start']:.3f}s–{det['end']:.3f}s, "
            f"{det['f_min']:.3f}–{det['f_max']:.3f} kHz)"
        )
        plt.tight_layout()
        plt.savefig(out_name, dpi=300)
        plt.close()


if __name__ == "__main__":
    AUDIO_PATH = "24.04.CTRL-F1 Day1.wav"
    CSV_PATH   = "plages.csv"

    # 1) charge les détections depuis le CSV (Hz -> kHz)
    detections = lire_detections_csv(CSV_PATH)

    # 2) trace et enregistre un PNG par détection dans ./detections/
    save_one_png_per_detection(
        AUDIO_PATH, detections,
        out_dir="detections",
        pad_t=0.3, pad_f=5.0,          # pad_f en kHz
        n_fft=4096, hop_length=512,
        target_sr=None                 # garde le SR d'origine (évite de perdre 60–80 kHz)
    )

    print(f"✅ Enregistré {len(detections)} images dans le dossier 'detections/'")
