import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow import keras
from playsound import playsound

app = Flask(__name__)

# Load your pre-trained model
model = keras.models.load_model('cnn.h5')

# Define the full path to the alert sound file
alert_sound_path = "C:\\Users\\dhilip\\Downloads\\siren-alert-96052.mp3"

# Function to detect fire in a frame
def detect_fire(frame):
    # Preprocess the frame (resize it to match your model's input size)
    frame = cv2.resize(frame, (64, 64))
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(frame)

    return prediction

def video_stream():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()  # Read a frame from the camera

        if not ret:
            continue

        # Detect fire in the frame
        result = detect_fire(frame)
        if result[0][0] > 0.5:
            cv2.putText(frame, 'Fire Detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if result[0][0] > 0.5:
            playsound(alert_sound_path)  # Play the alert sound when fire is detected

        # Provide feedback on the frame
        
        else:
            cv2.putText(frame, 'No Fire', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
