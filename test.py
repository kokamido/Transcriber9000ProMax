from transcriber9000.dto import AudioInput, TranscriberModel
from transcriber9000.transcriber_models import gigaam, tone

gigaam_transcriber = gigaam.Gigaam()
input = AudioInput(sample_rate=16000, source_audio_path="./data/test.mp3")
print(gigaam_transcriber.transcribe_single_audio(input, model=TranscriberModel.GIGAAM_V3_CTC, emit_timestamps=True))
print(gigaam_transcriber.transcribe_single_audio(input, model=TranscriberModel.GIGAAM_V3_RNNT, emit_timestamps=True))
print(gigaam_transcriber.transcribe_single_audio(input, model=TranscriberModel.GIGAAM_V3_E2E_CTC, emit_timestamps=True))
print(gigaam_transcriber.transcribe_single_audio(input, model=TranscriberModel.GIGAAM_V3_E2E_RNNT, emit_timestamps=True))

print(gigaam_transcriber.transcribe_single_audio(input, model=TranscriberModel.GIGAAM_V3_CTC, emit_timestamps=False))
print(gigaam_transcriber.transcribe_single_audio(input, model=TranscriberModel.GIGAAM_V3_RNNT, emit_timestamps=False))
print(gigaam_transcriber.transcribe_single_audio(input, model=TranscriberModel.GIGAAM_V3_E2E_CTC, emit_timestamps=False))
print(gigaam_transcriber.transcribe_single_audio(input, model=TranscriberModel.GIGAAM_V3_E2E_RNNT, emit_timestamps=False))

tone_transciber = tone.Tone()
input = AudioInput(sample_rate=8000, source_audio_path="./data/test.mp3")

print(tone_transciber.transcribe_single_audio(input, model=TranscriberModel.TONE_CTC, emit_timestamps=True))
print(tone_transciber.transcribe_single_audio(input, model=TranscriberModel.TONE_CTC, emit_timestamps=False))
