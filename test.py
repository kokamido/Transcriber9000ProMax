import gigaam
import numpy as np
from tone import StreamingCTCPipeline, read_audio

model_name = "v3_e2e_rnnt"
model = gigaam.load_model(model_name)
print(model.transcribe("/app/data/test.mp3", word_timestamps=True).words)

model_name = "v3_e2e_ctc"
model = gigaam.load_model(model_name)
print(model.transcribe("/app/data/test.mp3", word_timestamps=True).words)

model_name = "v3_rnnt"
model = gigaam.load_model(model_name)
print(model.transcribe("/app/data/test.mp3", word_timestamps=True).words)

model_name = "v3_ctc"
model = gigaam.load_model(model_name)
print(model.transcribe("/app/data/test.mp3", word_timestamps=True).words)

audio = read_audio("/app/data/test.mp3")

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
