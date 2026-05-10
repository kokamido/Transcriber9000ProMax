import gigaam


model_name = "v3_e2e_rnnt" 
model = gigaam.load_model(model_name)
print(model.transcribe('/app/data/test.mp3', word_timestamps=True))

model_name = "v3_e2e_ctc" 
model = gigaam.load_model(model_name)
print(model.transcribe('/app/data/test.mp3', word_timestamps=True))

model_name = "v3_rnnt" 
model = gigaam.load_model(model_name)
print(model.transcribe('/app/data/test.mp3', word_timestamps=True))

model_name = "v3_ctc" 
model = gigaam.load_model(model_name)
print(model.transcribe('/app/data/test.mp3', word_timestamps=True))


from tone import StreamingCTCPipeline, read_audio, read_example_audio
audio = read_audio('/app/data/test.mp3')

pipeline = StreamingCTCPipeline.from_hugging_face()
print(pipeline.forward_offline(audio))