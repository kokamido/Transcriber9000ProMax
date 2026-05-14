from typing import override

from gigaam import GigaAM, GigaAMASR, GigaAMEmo, load_model

from transcriber9000.dto import (
    AudioInput,
    TranscribedPhrase,
    TranscribedText,
    TranscriberModel,
)
from transcriber9000.logging import setup_logging

from .transcriber import ASRPipeline


class Gigaam(ASRPipeline):
    def __init__(self):
        super().__init__()
        self._model_cache: dict[TranscriberModel, GigaAM | GigaAMASR | GigaAMEmo] = {}
        self._logger = setup_logging().bind(vendor="gigaam")

    @override
    def transcribe(self, input: AudioInput, model: TranscriberModel, emit_timestamps: bool) -> TranscribedText:
        if input.sample_rate != 16000:
            self._logger.warning(
                f"Sample rate {input.sample_rate} does not supported by Gigaam. Audio will be resampled to 16kHz"
            )
        if model not in self._model_cache:
            match model:
                case TranscriberModel.GIGAAM_V3_CTC:
                    self._model_cache[model] = load_model("v3_ctc")
                case TranscriberModel.GIGAAM_V3_E2E_CTC:
                    self._model_cache[model] = load_model("v3_e2e_ctc")
                case TranscriberModel.GIGAAM_V3_RNNT:
                    self._model_cache[model] = load_model("v3_rnnt")
                case TranscriberModel.GIGAAM_V3_E2E_RNNT:
                    self._model_cache[model] = load_model("v3_e2e_rnnt")
                case _:
                    raise ValueError(f"Gigaam does not support {model=}")
        transcriber = self._model_cache[model]
        result = transcriber.transcribe(wav_file=input.source_audio_path, word_timestamps=emit_timestamps)
        return TranscribedText(
            source_path=input.source_audio_path,
            phrases=[TranscribedPhrase(start_time=w.start, end_time=w.end, text=w.text) for w in result.words],
        )
