from typing import final, override

from tone import StreamingCTCPipeline, read_audio

from transcriber9000.dto import AudioInput, TranscribedPhrase, TranscribedText, TranscriberModel
from transcriber9000.logging import setup_logging

from .transcriber import ASRPipeline


@final
class Tone(ASRPipeline):
    def __init__(self):
        super().__init__()
        self._pipeline = None
        self._logger = setup_logging().bind(vendor="tone")

    @override
    def transcribe_single_audio(
        self, input: AudioInput, model: TranscriberModel, emit_timestamps: bool
    ) -> TranscribedText | None:
        if input.sample_rate != 8000:
            self._logger.warning(
                f"Sample rate {input.sample_rate} does not supported by t-one. Audio will be resampled to 8kHz"
            )
        if model != TranscriberModel.TONE_CTC:
            raise ValueError(f"T-one supports only {TranscriberModel.TONE_CTC} model but {model=}")

        if self._pipeline is None:
            self._pipeline = StreamingCTCPipeline.from_hugging_face()  # todo get rid of HF

        audio = read_audio(input.source_audio_path)
        transcriber_result = self._pipeline.forward_offline(audio=audio)
        result = TranscribedText(
            model=model,
            source_path=input.source_audio_path,
            phrases=[
                TranscribedPhrase(
                    text=p.text,
                    start_time=p.start_time if emit_timestamps else None,
                    end_time=p.end_time if emit_timestamps else None,
                )
                for p in transcriber_result
            ],
        )
        return result
