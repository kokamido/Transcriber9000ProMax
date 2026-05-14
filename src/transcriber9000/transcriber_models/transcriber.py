from abc import ABC, abstractmethod
from collections.abc import Iterable

from transcriber9000.dto import AudioInput, TranscribedText, TranscriberModel


class ASRPipeline(ABC):
    @abstractmethod
    def transcribe(
        self, input: AudioInput, model: TranscriberModel
    ) -> TranscribedText: ...
    @abstractmethod
    def transcribe_batch(
        self, input: Iterable[AudioInput], model: TranscriberModel
    ) -> list[TranscribedText]: ...
