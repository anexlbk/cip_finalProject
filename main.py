import cv2
from fer import FER

# Emotion to emoji mapping
emotion_emoji = {
    'angry': '\-(>_<)-/',
    'disgust': '\-(>o<)-/',
    'fear': '\-(">_<")-/',
    'happy': '\-(>-<)-/',
    'neutral': '\-(>_<)-/',
    'sad': '\-(>_<)-/',
    'surprise': '\-(>_<)-/'
}

# Initialize the emotion detector
detector = FER()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    result = detector.detect_emotions(frame)

    for face in result:
        # Get bounding box
        (x, y, w, h) = face['box']

        # Get dominant emotion
        emotions = face['emotions']
        dominant_emotion = max(emotions, key=emotions.get)
        emoji = emotion_emoji.get(dominant_emotion, '😐')

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display emotion + emoji above the face
        label = f"{dominant_emotion} {emoji}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the window
    cv2.imshow('Emotion Detector', frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()