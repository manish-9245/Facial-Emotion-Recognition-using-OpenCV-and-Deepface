import cv2
import streamlit as st
from deepface import DeepFace
import tempfile

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def analyze_emotions(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y + h, x:x + w]

        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        emotion = result[0]['dominant_emotion']

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame

def main():
    st.title("Real-time Emotion Detection")
    st.write("Upload a video file or use your webcam for real-time emotion detection.")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    use_webcam = st.checkbox("Use webcam")

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video = cv2.VideoCapture(tfile.name)
        if not video.isOpened():
            st.error("Error: Could not open video file.")
            return

    elif use_webcam:
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            st.error("Error: Could not access webcam.")
            return

    else:
        st.write("Please upload a video file or select the webcam option.")
        return

    stframe = st.empty()
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = analyze_emotions(frame)
        stframe.image(frame, channels="BGR")

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# from PIL import Image