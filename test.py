import numpy as np
from tone import StreamingCTCPipeline, read_audio

from transcriber9000.dto import AudioInput, TranscriberModel
from transcriber9000.transcriber_models import gigaam

gigaam_transcriber = gigaam.Gigaam()
input = AudioInput(sample_rate=16000, source_audio_path="./data/test.mp3")
print(gigaam_transcriber.transcribe(input, model=TranscriberModel.GIGAAM_V3_CTC, emit_timestamps=True))
print(gigaam_transcriber.transcribe(input, model=TranscriberModel.GIGAAM_V3_RNNT, emit_timestamps=True))
print(gigaam_transcriber.transcribe(input, model=TranscriberModel.GIGAAM_V3_E2E_CTC, emit_timestamps=True))
print(gigaam_transcriber.transcribe(input, model=TranscriberModel.GIGAAM_V3_E2E_RNNT, emit_timestamps=True))

audio = read_audio("./data/test.mp3")

pipeline = StreamingCTCPipeline.from_hugging_face()
print(pipeline.forward_offline(audio))

CHUNK_SIZE = 2400
audio = np.pad(audio, (0, -len(audio) % CHUNK_SIZE))
state = None

for i in range(0, len(audio) - CHUNK_SIZE, CHUNK_SIZE):
    out, state = pipeline.forward(audio[i : i + CHUNK_SIZE], state=state)
    if out:
        print(out)
out, state = pipeline.forward(audio[i : i + CHUNK_SIZE], state=state, is_last=True)
if out:
    print(out)
