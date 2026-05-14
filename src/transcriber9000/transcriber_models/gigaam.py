from dataclasses import asdict
from typing import override

from gigaam import GigaAMASR, load_model
from loguru import Logger

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
        self._model_cache: dict[TranscriberModel, GigaAMASR] = {}
        self._logger: Logger = setup_logging().bind(vendor="gigaam")

    def __load_asr_model(self, model: TranscriberModel) -> GigaAMASR:
        res = None
        match model:
            case TranscriberModel.GIGAAM_V3_CTC:
                res = load_model("v3_ctc")
            case TranscriberModel.GIGAAM_V3_E2E_CTC:
                res = load_model("v3_e2e_ctc")
            case TranscriberModel.GIGAAM_V3_RNNT:
                res = load_model("v3_rnnt")
            case TranscriberModel.GIGAAM_V3_E2E_RNNT:
                res = load_model("v3_e2e_rnnt")
            case _:
                raise ValueError(f"Gigaam does not support {model=}")
        if not isinstance(res, GigaAMASR):
            raise TypeError("Can't load model of type GigaAMASR")
        return res

    @override
    def transcribe(self, input: AudioInput, model: TranscriberModel, emit_timestamps: bool) -> TranscribedText | None:
        if input.sample_rate != 16000:
            self._logger.warning(
                f"Sample rate {input.sample_rate} does not supported by Gigaam. Audio will be resampled to 16kHz"
            )
        if model not in self._model_cache:
            self._model_cache[model] = self.__load_asr_model(model)
        transcriber = self._model_cache[model]
        with self._logger.contextualize(model=model, **input.model_dump(mode="json")):
            transcription_result = transcriber.transcribe(
                wav_file=str(input.source_audio_path), word_timestamps=emit_timestamps
            )
            if transcription_result.words is not None:
                self._logger.info("Transcribed")
                self._logger.debug(f"transcription_result: {asdict(transcription_result)}")
                result = TranscribedText(
                    source_path=input.source_audio_path,
                    phrases=[
                        TranscribedPhrase(start_time=w.start, end_time=w.end, text=w.text) for w in transcription_result.words
                    ],
                )
                self._logger.debug(f"result: {result.model_dump_json()}")
                return result
            self._logger.error("transcription_result.words is None.")
