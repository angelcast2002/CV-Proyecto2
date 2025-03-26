import cv2
import time
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
from mediapipe.framework.formats import landmark_pb2
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

latest_annotated_frame = None

def draw_landmarks_on_image(image_mp: mp.Image, hand_landmarks):
    annotated_image = cv2.cvtColor(image_mp.numpy_view(), cv2.COLOR_RGB2BGR)

    for single_hand_lm in hand_landmarks:
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        for lmk in single_hand_lm:
            landmark_list.landmark.add(x=lmk.x, y=lmk.y, z=lmk.z)

        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=landmark_list,
            connections=HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=3
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2
            ),
        )
    return annotated_image

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_annotated_frame

    
    if result.hand_landmarks:
        annotated = draw_landmarks_on_image(output_image, result.hand_landmarks)
    else:
        
        annotated = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR)

    latest_annotated_frame = annotated

def main():
    model_path = "./hand_landmarker.task"

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_hands=2
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame de la cámara.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(time.time() * 1000)

            landmarker.detect_async(mp_image, timestamp_ms)

            if latest_annotated_frame is not None:
                cv2.imshow("HandLandmarker Live", latest_annotated_frame)
            else:
                cv2.imshow("HandLandmarker Live", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
