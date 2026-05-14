from typing import override

from transcriber9000.dto import AudioInput, TranscribedText, TranscriberModel

from .transcriber import ASRPipeline


class Tone(ASRPipeline):
    @override
    def transcribe(self, input: AudioInput, model: TranscriberModel, emit_timestamps: bool) -> TranscribedText | None:
        raise NotImplementedError
