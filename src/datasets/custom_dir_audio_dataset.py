from pathlib import Path
from typing import Dict, Any, List, Optional

import torchaudio

from src.datasets.base_dataset import BaseDataset


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}


def _get_audio_len_sec(path: Path) -> float:
    try:
        info = torchaudio.info(str(path))
        if info.num_frames is not None and info.sample_rate and info.sample_rate > 0:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass

    wav, sr = torchaudio.load(str(path))
    return float(wav.shape[-1]) / float(sr)


class CustomDirAudioDataset(BaseDataset):
    def __init__(
        self,
        audio_dir: str,
        transcription_dir: Optional[str] = None,
        *args, **kwargs
    ):
        audio_dir = Path(audio_dir)
        trans_dir = Path(transcription_dir) if transcription_dir else None

        data: List[Dict[str, Any]] = []
        for path in sorted(audio_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in AUDIO_EXTS:
                continue

            entry: Dict[str, Any] = {
                "path": str(path),
                "audio_len": _get_audio_len_sec(path),
            }

            text = ""
            if trans_dir and trans_dir.exists():
                tpath = trans_dir / (path.stem + ".txt")
                if tpath.exists():
                    try:
                        text = tpath.read_text(encoding="utf-8").strip()
                    except UnicodeDecodeError:
                        text = tpath.read_text(errors="ignore").strip()
            entry["text"] = text
            data.append(entry)
        super().__init__(data, *args, **kwargs)
