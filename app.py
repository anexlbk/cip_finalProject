import gradio as gr
import os
from deepface import DeepFace

emotion_emoji = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😄', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

def detect_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
        dominant = result[0]['dominant_emotion']
        return f"{dominant} {emotion_emoji.get(dominant, '😐')}"
    except Exception:
        return "No face detected"

demo = gr.Interface(
    fn=detect_emotion,
    inputs=gr.Image(sources="webcam", type="numpy"),
    outputs="text",
    title="Real-Time Emotion Detector with DeepFace",
    description="Point your webcam at your face to detect emotions using DeepFace"
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))