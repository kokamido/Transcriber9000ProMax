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
from tone import StreamingCTCPipeline, read_audio

from transcriber9000.dto import TranscriberModel

if TYPE_CHECKING:
    from numpy import int32
    from numpy.typing import NDArray
    from torch import Tensor


def get_model(
    model: TranscriberModel,
) -> GigaAM | GigaAMEmo | GigaAMASR | StreamingCTCPipeline:
    match model:
        case TranscriberModel.GIGAAM_V3_CTC:
            return load_model("v3_ctc")
        case TranscriberModel.GIGAAM_V3_RNNT:
            return load_model("v3_rnnt")
        case TranscriberModel.GIGAAM_V3_E2E_CTC:
            return load_model("v3_e2e_ctc")
        case TranscriberModel.GIGAAM_V3_E2E_RNNT:
            return load_model("v3_e2e_rnnt")
        case TranscriberModel.TONE_CTC:
            return StreamingCTCPipeline.from_hugging_face()
        case _:
            raise ValueError(f"Can't load {model=}")


def get_audio_loader(
    model: TranscriberModel, sample_rate: int | None
) -> Callable[[str], Tensor | NDArray[int32]]:
    match model:
        case (
            TranscriberModel.GIGAAM_V3_CTC
            | TranscriberModel.GIGAAM_V3_E2E_CTC
            | TranscriberModel.GIGAAM_V3_RNNT
            | TranscriberModel.GIGAAM_V3_E2E_RNNT
        ):
            if sample_rate is None:
                raise ValueError(f"You must set sample_rate for {model=}")
            return lambda audio_path: load_audio(
                audio_path=audio_path, sample_rate=sample_rate
            )
        case TranscriberModel.TONE_CTC:
            if sample_rate is not None and sample_rate != 8000:
                raise ValueError(f"t-one supports 8kHz only, not {sample_rate=}")
            return read_audio
