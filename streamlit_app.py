import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2
from deepface import DeepFace
import tempfile
import time
import av

st.title("Real-time Emotion Detection")

# Load the pre-trained Haar Cascade face detection model
haar_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

class EmotionDetectionProcessor(VideoProcessorBase):
    """Class to process video frames and detect emotions."""

    def __init__(self):
        self.face_cascade = face_cascade

    def recv(self, frame):
        """Process each frame from the webcam stream."""
        img = frame.to_ndarray(format="bgr24")

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = img[y:y + h, x:x + w]

            # Analyze emotion of the detected face
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            # Draw rectangle and emotion label around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def analyze_emotions_in_video(uploaded_file):
    """Analyze emotions in an uploaded video file."""
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video = cv2.VideoCapture(tfile.name)
    
    stframe = st.empty()
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Analyze emotion of the detected face
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            # Draw rectangle and emotion label around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Display the frame with detected emotions
        stframe.image(frame, channels="BGR")
    
    video.release()
    cv2.destroyAllWindows()

# UI to select between webcam and file upload
mode = st.selectbox("Select Input Mode", ["Webcam", "Upload Video File"])

if mode == "Webcam":
    # WebRTC configuration
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    # Start the webcam stream
    webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=EmotionDetectionProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True,
    )

elif mode == "Upload Video File":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        analyze_emotions_in_video(uploaded_file)
