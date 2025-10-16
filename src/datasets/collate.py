import torch

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    """
    # --------- gather ----------
    audios = []
    audios_aug = []
    spectrograms = []
    texts = []
    text_encoded_list = []
    audio_paths = []

    for it in dataset_items:
        # original audio -> [1, T]
        a = it["audio"]
        if a.dim() == 1:
            a = a.unsqueeze(0)
        audios.append(a)

        # augmented audio (fallback to original if not present)
        aa = it.get("audio_aug", a)
        if aa.dim() == 1:
            aa = aa.unsqueeze(0)
        audios_aug.append(aa)

        # spectrogram -> [F, T]
        s = it["spectrogram"]
        if s.dim() == 3 and s.size(0) == 1:  # [1, F, T]
            s = s.squeeze(0)
        elif s.dim() != 2:
            raise ValueError(f"Unexpected spectrogram shape: {tuple(s.shape)}")
        spectrograms.append(s)

        texts.append(it["text"])

        te = it["text_encoded"]
        if not isinstance(te, torch.Tensor):
            te = torch.tensor(te, dtype=torch.long)
        else:
            te = te.to(dtype=torch.long)
        text_encoded_list.append(te)

        audio_paths.append(it["audio_path"])

    # --------- lengths ----------
    audio_lengths = torch.as_tensor([a.shape[-1] for a in audios], dtype=torch.long)
    audio_aug_lengths = torch.as_tensor([a.shape[-1] for a in audios_aug], dtype=torch.long)
    spec_lengths = torch.as_tensor([s.shape[-1] for s in spectrograms], dtype=torch.long)
    text_encoded_lengths = torch.as_tensor([t.numel() for t in text_encoded_list], dtype=torch.long)

    B = len(audios)

    # --------- pad original audio ----------
    max_audio_len = int(audio_lengths.max().item())
    padded_audios = audios[0].new_zeros((B, 1, max_audio_len))
    for i, a in enumerate(audios):
        T = a.shape[-1]
        padded_audios[i, :, :T] = a

    # --------- pad augmented audio ----------
    max_audio_aug_len = int(audio_aug_lengths.max().item())
    padded_audios_aug = audios_aug[0].new_zeros((B, 1, max_audio_aug_len))
    for i, a in enumerate(audios_aug):
        T = a.shape[-1]
        padded_audios_aug[i, :, :T] = a

    # --------- pad spectrograms (pad along time) ----------
    F = spectrograms[0].shape[0]
    for s in spectrograms:
        if s.shape[0] != F:
            raise ValueError("Inconsistent number of frequency bins across spectrograms in a batch.")
    max_spec_len = int(spec_lengths.max().item())
    FLOOR_DB = -80.0
    padded_specs = torch.full(
        (B, F, max_spec_len),
        fill_value=FLOOR_DB,
        dtype=spectrograms[0].dtype,
        device=spectrograms[0].device,
    )
    for i, s in enumerate(spectrograms):
        T = s.shape[-1]
        padded_specs[i, :, :T] = s.clamp_min(FLOOR_DB)

    # --------- pad text_encoded (CTC targets) ----------
    pad_value = 0
    max_txt_len = int(text_encoded_lengths.max().item())
    padded_text_encoded = torch.full((B, max_txt_len), pad_value, dtype=torch.long)
    for i, t in enumerate(text_encoded_list):
        L = t.numel()
        padded_text_encoded[i, :L] = t

    # --------- pack result ----------
    result_batch = {
        "audio": padded_audios_aug,              # [B, 1, T_aug_max]
        "audio_length": audio_aug_lengths,       # [B]
        "spectrogram": padded_specs,                 # [B, F, S_max]
        "spectrogram_length": spec_lengths,          # [B]
        "text": texts,                               # list[str]
        "text_encoded": padded_text_encoded,         # [B, L_max] (padded with -1)
        "text_encoded_length": text_encoded_lengths, # [B]
        "audio_path": audio_paths,                   # list[str]
    }
    return result_batch
