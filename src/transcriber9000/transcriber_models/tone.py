from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from tone import StreamingCTCPipeline, read_audio

from transcriber9000.dto import TranscriberModel

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt


def load_tone(model: TranscriberModel) -> StreamingCTCPipeline:
    match model:
        case TranscriberModel.TONE_CTC:
            return StreamingCTCPipeline.from_hugging_face()
        case _:
            raise ValueError(f"Can't load {model=}")


def get_audio_loader() -> Callable[[Path], npt.NDArray[np.int32]]:
    return read_audio
