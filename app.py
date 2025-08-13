import streamlit as st
import whisper
import tempfile
import numpy as np
import os
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings

# ---------------------------
# Load Whisper Model (small only)
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    st.write("Loading Whisper model: **small**")
    return whisper.load_model("small")

model = load_model()

# ---------------------------
# Audio Processor for mic input
# ---------------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.buffer = b""

    def recv_audio(self, frame):
        self.buffer += frame.to_ndarray().tobytes()

    def get_audio_buffer(self):
        return self.buffer

# ---------------------------
# Transcription helper
# ---------------------------
def transcribe_audio(file_path):
    audio = whisper.load_audio(file_path)       # mono, 16kHz
    audio = whisper.pad_or_trim(audio)

    # sanity check
    if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.size == 0:
        return "Invalid or empty audio file."

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    return f"Detected language: {detected_lang}\n\n{result.text}"

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎙️ Whisper Speech-to-Text (small model)")

mode = st.radio("Choose input method:", ("🎤 Record from mic", "📂 Upload audio file"))

if mode == "🎤 Record from mic":
    ctx = webrtc_streamer(
        key="whisper-mic",
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False}
        ),
        audio_processor_factory=AudioProcessor,
    )

    if ctx.audio_processor:
        if st.button("Stop & Transcribe"):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with open(tmp_file.name, "wb") as f:
                f.write(ctx.audio_processor.get_audio_buffer())

            with st.spinner("Transcribing..."):
                text = transcribe_audio(tmp_file.name)
            st.success("Done!")
            st.text_area("Transcription", text, height=200)

elif mode == "📂 Upload audio file":
    uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
    if uploaded_file is not None:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_file.write(uploaded_file.read())
        tmp_file.flush()

        with st.spinner("Transcribing..."):
            text = transcribe_audio(tmp_file.name)
        st.success("Done!")
        st.text_area("Transcription", text, height=200)
