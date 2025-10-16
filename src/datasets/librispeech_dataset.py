import json
import os
import shutil
from pathlib import Path

import torchaudio
import wget
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == "train-clean" or part == "train_all"

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        else:
            data_dir = Path(data_dir)
        self._data_dir = data_dir

        self.kaggle = False
        if 'kaggle' in str(self._data_dir).lower():
            self.kaggle = True

        self.kaggle_download = kwargs.pop("kaggle_download", False)
        self.kaggle_work_dir = Path(kwargs.pop("kaggle_work_dir", "/kaggle/working/librispeech"))

        if part == "train_all":
            index = sum(
                [
                    self._get_or_load_index(part)
                    for part in URL_LINKS
                    if "train" in part
                ],
                [],
            )
        elif part == "train-clean":
            index = sum(
                [
                    self._get_or_load_index(part)
                    for part in URL_LINKS
                    if "train-clean" in part
                ],
                [],
            )
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)


    def _load_part(self, part, base_dir: Path = None):
        if base_dir is None:
            base_dir = self._data_dir
        base_dir.mkdir(exist_ok=True, parents=True)

        arch_path = base_dir / f"{part}.tar.gz"
        print(f"Loading part {part} -> {arch_path}")
        wget.download(URL_LINKS[part], str(arch_path))
        shutil.unpack_archive(arch_path, base_dir)
        for fpath in (base_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(base_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(base_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        if not self.kaggle:
            index_path = self._data_dir / f"{part}_index.json"
            if index_path.exists():
                with index_path.open() as f:
                    index = json.load(f)
            else:
                index = self._create_index(part)
                with index_path.open("w") as f:
                    json.dump(index, f, indent=2)
        else:
            Path("/kaggle/working/ConformerCTC/kaggle_data_index").mkdir(exist_ok=True, parents=True)
            index_path = Path("/kaggle/working/ConformerCTC/kaggle_data_index") / f"{part}_index.json"
            if index_path.exists():
                with index_path.open() as f:
                    index = json.load(f)
            else:
                index = self._create_index(part)
                with index_path.open("w") as f:
                    json.dump(index, f, indent=2)
        return index

    @staticmethod
    def _has_any_flac(root: Path) -> bool:
        if not root.exists():
            return False
        for _, _, filenames in os.walk(str(root)):
            if any(fn.endswith(".flac") for fn in filenames):
                return True
        return False

    def _create_index(self, part):
        index = []

        if self.kaggle:
            if self.kaggle_download:
                split_dir = self.kaggle_work_dir / part  
                if not self._has_any_flac(split_dir):
                    print(f"[Kaggle] Download mode ON -> {self.kaggle_work_dir}")
                    self._load_part(part, base_dir=self.kaggle_work_dir)
            else:
                split_dir = self._data_dir / part / "LibriSpeech" / part
        else:
            split_dir = self._data_dir / part
            if not split_dir.exists() or not self._has_any_flac(split_dir):
                self._load_part(part, base_dir=self._data_dir)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
            list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index
