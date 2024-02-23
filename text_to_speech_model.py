import numpy as np
import torch
import scipy
from transformers import VitsTokenizer, VitsModel
from io import BytesIO


# Generate raw audio data (16KHz sampling rate)
def load_tts_model():
    tokenizer = VitsTokenizer.from_pretrained('facebook/mms-tts-eng')
    model = VitsModel.from_pretrained('facebook/mms-tts-eng')

    return model, tokenizer


def text_to_speech(text, model, tokenizer):
    if not text:
        text = 'Hi from MMS TTS!'

    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        data = model(**inputs).waveform.cpu().numpy()

    # Convert to wav format
    buffer = BytesIO()

    data_i16 = (data * np.iinfo(np.int16).max).astype(np.int16)
    scipy.io.wavfile.write(buffer, rate=16000, data=data_i16.squeeze())

    data_wav = buffer.getbuffer().tobytes()

    return data_wav
