
import face_recognition

image = face_recognition.load_image_file("person.jpg")
encoding = face_recognition.face_encodings(image)[0]  # 128-d vector
