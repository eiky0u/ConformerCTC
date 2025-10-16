import logging
import random
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index: List[Dict[str, Any]],
        text_encoder,                   # WhisperBPEEncoder
        target_sr: int = 16000,
        limit: Optional[int] = None,
        max_audio_length: Optional[int] = None,
        max_text_length: Optional[int] = None,
        shuffle_index: bool = False,
        instance_transforms: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)

        self.text_encoder = text_encoder           
        index = self._filter_records_from_dataset(
            index=index,
            max_audio_length=max_audio_length,
            max_text_length=max_text_length,
            text_encoder=self.text_encoder,
        )

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index: List[Dict[str, Any]] = index
        self.target_sr = target_sr
        self.instance_transforms = instance_transforms

    def __getitem__(self, ind: int) -> Dict[str, Any]:
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio = self.load_audio(audio_path)

        text = data_dict["text"]
        text_encoded = self.text_encoder.encode(text)        

        if self.instance_transforms is not None and "get_wav_augs" in self.instance_transforms:
            audio_aug = self.get_wav_augs(audio)
        else:
            audio_aug = audio

        if self.instance_transforms is None or "get_spectrogram" not in self.instance_transforms:
            raise KeyError("instance_transforms must contain 'get_spectrogram' callable")
        spectrogram = self.get_spectrogram(audio_aug)

        instance_data = {
            "audio": audio,
            "audio_aug": audio_aug,
            "spectrogram": spectrogram,
            "text": text,
            "text_encoded": text_encoded,   
            "audio_path": audio_path,
        }

        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def __len__(self) -> int:
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)


    def load_audio(self, path: str) -> torch.Tensor:
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # mono
        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.target_sr)
        return audio_tensor

    def get_wav_augs(self, audio: torch.Tensor) -> torch.Tensor:
        return self.instance_transforms["get_wav_augs"](audio)

    def get_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        return self.instance_transforms["get_spectrogram"](audio)

    def preprocess_data(self, instance_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.instance_transforms is not None:
            for k, transform in self.instance_transforms.items():
                if k in instance_data:
                    instance_data[k] = transform(instance_data[k])
        return instance_data



    @staticmethod
    def _filter_records_from_dataset(
        index: List[Dict[str, Any]],
        max_audio_length: Optional[int],
        max_text_length: Optional[int],
        text_encoder,
    ) -> List[Dict[str, Any]]:
        """
        Filter some of the elements from the dataset depending on
        some condition.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        initial_size = len(index)

        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["audio_len"] for el in index]) >= max_audio_length
            n_drop = int(exceeds_audio_length.sum())
            if n_drop > 0:
                logger.info(f"{n_drop} ({n_drop / initial_size:.1%}) records are longer than "
                            f"{max_audio_length} sec. Excluding them.")
        else:
            exceeds_audio_length = False

        if max_text_length is not None:
            norm_lens = np.array([len(text_encoder.normalize_text(el["text"])) for el in index])
            exceeds_text_length = norm_lens >= max_text_length
            n_drop_t = int(exceeds_text_length.sum())
            if n_drop_t > 0:
                logger.info(f"{n_drop_t} ({n_drop_t / initial_size:.1%}) records are longer than "
                            f"{max_text_length} chars (normalized). Excluding them.")
        else:
            exceeds_text_length = False

        mask_drop = exceeds_audio_length | exceeds_text_length
        if mask_drop is not False and np.any(mask_drop):
            kept = [el for el, drop in zip(index, mask_drop) if not drop]
            logger.info(f"Filtered {int(np.sum(mask_drop))} "
                        f"({np.sum(mask_drop) / initial_size:.1%}) records from dataset.")
            return kept

        return index

    @staticmethod
    def _assert_index_is_valid(index: List[Dict[str, Any]]) -> None:
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "path" in entry, "Each dataset item must include 'path' (path to audio file)."
            assert "text" in entry, "Each dataset item must include 'text' (ground-truth transcription)."
            assert "audio_len" in entry, "Each dataset item must include 'audio_len' (audio length in sec)."

    @staticmethod
    def _sort_index(index: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(index, key=lambda x: x["audio_len"])

    @staticmethod
    def _shuffle_and_limit_index(
        index: List[Dict[str, Any]],
        limit: Optional[int],
        shuffle_index: bool,
    ) -> List[Dict[str, Any]]:
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)
        if limit is not None:
            index = index[:limit]
        return index
