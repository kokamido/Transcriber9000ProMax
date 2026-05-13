from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from gigaam import (  # pyright: ignore[reportMissingTypeStubs]
    GigaAM,
    GigaAMASR,
    GigaAMEmo,
    load_audio,
    load_model,
)

from transcriber9000.dto import TranscriberModel

if TYPE_CHECKING:
    from torch import Tensor


def load_gigaam(model: TranscriberModel) -> GigaAM | GigaAMEmo | GigaAMASR:
    match model:
        case TranscriberModel.GIGAAM_V3_CTC:
            return load_model("v3_ctc")
        case TranscriberModel.GIGAAM_V3_RNNT:
            return load_model("v3_rnnt")
        case TranscriberModel.GIGAAM_V3_E2E_CTC:
            return load_model("v3_e2e_ctc")
        case TranscriberModel.GIGAAM_V3_E2E_RNNT:
            return load_model("v3_e2e_rnnt")
        case _:
            raise ValueError(f"Can't load {model=}")


def get_audio_loader(sample_rate: int) -> Callable[[str], Tensor]:
    return lambda audio_path: load_audio(audio_path=audio_path, sample_rate=sample_rate)
