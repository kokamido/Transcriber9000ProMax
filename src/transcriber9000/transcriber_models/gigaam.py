from collections.abc import Iterable
from typing import override

from transcriber9000.dto import AudioInput, TranscribedText, TranscriberModel

from .transcriber import ASRPipeline


class Gigaam(ASRPipeline):
    @override
    def transcribe(self, input: AudioInput, model: TranscriberModel) -> TranscribedText:
        raise NotImplementedError

    @override
    def transcribe_batch(
        self, input: Iterable[AudioInput], model: TranscriberModel
    ) -> list[TranscribedText]:
        raise NotImplementedError
