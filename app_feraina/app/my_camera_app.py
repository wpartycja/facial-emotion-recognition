from flask import Flask, render_template, Response, request
import cv2
from PIL import Image
import requests
import io

app = Flask(__name__, template_folder='./templates')
global camera
camera = None

global switch
switch = 0

haar_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                      "haarcascade_frontalface_default.xml")
# cnn_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
model_url = "http://localhost:8000/prediction"


def detect_emotion(frame):
    # może być do zmiany
    image = Image.fromarray(frame)
    buf = io.BytesIO()
    image.save(buf, format='png')
    byte_im = buf.getvalue()
    files = {'file': ("some_useful_name", byte_im, "image/jpeg")}
    predicted_emotion = requests.post(model_url, files=files).json()["emotion"]
    return predicted_emotion


def detect_faces(detector, frame):
    rects = detector.detectMultiScale(frame,
                                      scaleFactor=1.05,
                                      minNeighbors=5,
                                      minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    return rects


def detect_faces_cnn(detector, frame):
    return detector(frame, 1)


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    count = 0
    while True:
        success, frame = camera.read()
        if success:
            try:
                # default, clean frame
                frame = cv2.flip(frame, 1)

                if count % 2 == 0:
                
                    # face detection
                    rects = detect_faces(haar_detector, frame)
                
                    # ẹmotion prediction
                    preds = []
                    for (x, y, w, h) in rects:
                        lil_frame = frame[y:y+h, x:x+w]
                        pred = detect_emotion(lil_frame)
                        preds.append(pred)
                        # print(pred)

                # drawing bboxes and writing labels
                for idx, (x, y, w, h) in enumerate(rects):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (133,51,255), 2)   # (178,102,255)
                    cv2.putText(frame, preds[idx], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (133,51,255), 2, cv2.LINE_AA)

                # creating frame with all this stuff
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                count += 1
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            except Exception as e:
                pass

        else:
            pass


@app.route('/')
def index():
    return render_template('camera_permission.html')

@app.route('/go')
def go():
    global camera
    camera = cv2.VideoCapture(0)
    return render_template('index1.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('stop') == 'Stop':
            # if switch == 1:
            switch = 0
            camera.release()
            cv2.destroyAllWindows()
        else:
            camera = cv2.VideoCapture(0)
            switch = 1
    elif request.method == 'GET':
        return render_template('index1.html')
    return render_template('index1.html')


if __name__ == '__main__':
    app.run()
