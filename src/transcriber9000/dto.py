from enum import StrEnum
from pathlib import Path

from pydantic import (
    BaseModel,
    FilePath,
    NonNegativeFloat,
    PositiveInt,
    StrictStr,
    model_validator,
)


class TranscriberModel(StrEnum):
    GIGAAM_V3_E2E_CTC = "GIGAAM_V3_E2E_CTC"
    GIGAAM_V3_E2E_RNNT = "GIGAAM_V3_E2E_RNNT"
    GIGAAM_V3_CTC = "GIGAAM_V3_CTC"
    GIGAAM_V3_RNNT = "GIGAAM_V3_RNNT"
    TONE_CTC = "TONE_CTC"


class Preprocessor(StrEnum):
    SPLIT_BY_VAD = "SPLIT_BY_VAD"


class TranscriberConfig(BaseModel):
    transcriber_model: TranscriberModel
    preprocessors: list[Preprocessor]


class AudioInput(BaseModel):
    source_audio_path: FilePath
    sample_rate: PositiveInt


class TimeInterval(BaseModel):
    start_time: NonNegativeFloat
    end_time: NonNegativeFloat

    @model_validator(mode="after")
    def check_time_order(self) -> "TimeInterval":
        if self.start_time > self.end_time:
            raise ValueError("start_time must be <= end_time")
        return self


class TranscribedPhrase(TimeInterval):
    text: str


class TranscribedText(BaseModel):
    phrases: list[TranscribedPhrase]
    source_path: Path


class DrawableEvent(TimeInterval):
    label: StrictStr
    inner_text: str
