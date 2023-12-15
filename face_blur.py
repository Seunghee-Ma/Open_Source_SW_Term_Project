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

if __name__ == "__main__":
    # 사용자로부터 동영상 또는 이미지 경로 입력 받기
    file_path = input("Enter the path of video or image: ")

    # 파일이 동영상인지 이미지인지 확인 후 처리
    if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(file_path)
    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        process_image(file_path)
    else:
        print("Unsupported file format. Please provide a video or image file.")
