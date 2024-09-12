from flask import Flask, request, jsonify
import cv2
import face_recognition
import numpy as np
import base64

app = Flask(__name__)

# Load known face images
known_image = face_recognition.load_image_file("known_person.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]
known_face_names = ["Person 1"]

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    image_data = data['image'].split(',')[1]
    decoded_image = base64.b64decode(image_data)
    nparr = np.frombuffer(decoded_image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect and recognize faces
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    name = "Unknown"
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        if True in matches:
            name = known_face_names[matches.index(True)]

    return jsonify({"name": name})

if __name__ == '__main__':
    app.run(debug=True)