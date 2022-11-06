import face_recognition
import cv2
import numpy as np
import os
import time
from pynput import keyboard 
from pynput.keyboard import Key, Controller

video_capture = cv2.VideoCapture(0)

petko_image = face_recognition.load_image_file("petko.jpeg")
petko_face_encoding = face_recognition.face_encodings(petko_image)[0]

svetla_image = face_recognition.load_image_file("svetla.jpeg")
svetla_face_encoding = face_recognition.face_encodings(svetla_image)[0]

known_face_encodings = [
    petko_face_encoding,
    svetla_face_encoding
]
known_face_names = [
    "Petko Tankov",
    "Svetla Tankova"
]

isSentNotif = False

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
face_recognized = False

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if (name=="Petko Tankov") and (isSentNotif==False):
                    isSentNotif = True
                    os.system("open -a Messages")
                    time. sleep (2)
                    keyboard = Controller ()
                    keyboard. type("Петко се прибра")
                    time. sleep(1)
                    keyboard.press (Key.enter)
                
            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 3
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 10, bottom - 10), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
