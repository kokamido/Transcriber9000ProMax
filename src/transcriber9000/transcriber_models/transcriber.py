from abc import ABC, abstractmethod

from transcriber9000.dto import AudioInput, TranscribedText, TranscriberModel


class ASRPipeline(ABC):
    @abstractmethod
    def transcribe_single_audio(
        self, input: AudioInput, model: TranscriberModel, emit_timestamps: bool
    ) -> TranscribedText | None: ...
