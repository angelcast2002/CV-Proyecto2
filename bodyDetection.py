import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# === CONFIGURACIÓN ===
model_path = "pose_landmarker_heavy.task"
num_poses = 6
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5
frame_pose = None
frame_black = None

# === ELECCIÓN DE FUENTE ===
opcion = input("¿Qué quieres usar? (1 = Cámara, 2 = Archivo de video, 3 = Imagen): ")
usar_camara = False
usar_imagen = False

if opcion == '1':
    video_source = 0
    usar_camara = True
elif opcion == '2':
    video_path = input("Ruta del archivo de video (ej. baile.mp4): ")
    video_source = video_path
elif opcion == '3':
    imagen_path = input("Ruta de la imagen (ej. imagen.png): ")
    usar_imagen = True
else:
    print("Opción no válida.")
    exit()

# === FUNCIONES ===
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    black_image = np.zeros_like(rgb_image)

    for pose_landmarks in pose_landmarks_list:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=lmk.x, y=lmk.y, z=lmk.z) for lmk in pose_landmarks
        ])
        # Pose sobre imagen original
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
        # Pose sobre fondo negro
        mp.solutions.drawing_utils.draw_landmarks(
            black_image,
            proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())

    return annotated_image, black_image

# === CONFIGURACIÓN DEL MODELO ===
base_options = python.BaseOptions(model_asset_path=model_path)

if usar_camara:
    running_mode = vision.RunningMode.LIVE_STREAM

    def print_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        global frame_pose, frame_black
        try:
            rgb_image = output_image.numpy_view()
            annotated, black = draw_landmarks_on_image(rgb_image, result)
            frame_pose = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            frame_black = cv2.cvtColor(black, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error en callback: {e}")

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=print_result
    )

elif usar_imagen:
    print("Detectando en imagen...")
    running_mode = vision.RunningMode.IMAGE
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

else:
    running_mode = vision.RunningMode.VIDEO
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

# === PROCESAMIENTO ===
with vision.PoseLandmarker.create_from_options(options) as landmarker:

    # Modo imagen
    if usar_imagen:
        bgr = cv2.imread(imagen_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        annotated, black = draw_landmarks_on_image(rgb, result)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        black_bgr = cv2.cvtColor(black, cv2.COLOR_RGB2BGR)

        cv2.imshow("Imagen - Pose", annotated_bgr)
        cv2.imshow("Imagen - Fondo negro", black_bgr)
        cv2.imwrite("data/output/imagen_con_pose.png", annotated_bgr)
        cv2.imwrite("data/output/imagen_fondo_negro.png", black_bgr)
        print(f"Personas detectadas: {len(result.pose_landmarks)}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

    # Modo cámara o video
    cap = cv2.VideoCapture(video_source)
    frame_count = 0
    prev_time = time.time()

    if not usar_camara:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out_pose = cv2.VideoWriter('data/output/video_con_pose.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        out_skeleton = cv2.VideoWriter('data/output/video_fondo_negro.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        curr_time = time.time()
        fps_live = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        if usar_camara:
            landmarker.detect_async(mp_image, timestamp_ms)
            if frame_pose is not None and frame_black is not None:
                cv2.putText(frame_pose, f"Frame: {frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_pose, f"FPS: {fps_live:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(frame_black, f"Frame: {frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame_black, f"FPS: {fps_live:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("Pose en vivo", frame_pose)
                cv2.imshow("Esqueletos en vivo", frame_black)
        else:
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            annotated, black = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            black_bgr = cv2.cvtColor(black, cv2.COLOR_RGB2BGR)

            cv2.putText(annotated_bgr, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_bgr, f"FPS: {fps_live:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(black_bgr, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(black_bgr, f"FPS: {fps_live:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out_pose.write(annotated_bgr)
            out_skeleton.write(black_bgr)

            cv2.imshow("Video - Pose", annotated_bgr)
            cv2.imshow("Video - Fondo negro", black_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if not usar_camara:
        out_pose.release()
        out_skeleton.release()
    cv2.destroyAllWindows()
