import cv2

def apply_face_mosaic(image):
    # 얼굴 감지기 초기화
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # 감지된 얼굴에 모자이크 적용
    for (x, y, w, h) in faces:
        # 얼굴 부분 추출
        face_roi = image[y:y+h, x:x+w]

        # 모자이크 처리
        face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)

        # 모자이크를 원본 이미지에 적용
        image[y:y+face_roi.shape[0], x:x+face_roi.shape[1]] = face_roi

    return image
