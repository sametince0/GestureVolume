import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# MediaPipe el izleme modülü
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Kamera başlatma
cap = cv2.VideoCapture(0)

# Pycaw kullanarak ses kontrolü ayarları
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Ses seviyesinin min ve max değerleri
vol_min, vol_max = volume.GetVolumeRange()[:2]
previous_volume = vol_max  # Son ses seviyesi

# OpenCV Haar Cascade yüz algılama modeli yükleme
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Gri tonlama (yüz algılama için)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))


    # BGR'den RGB'ye dönüştürme
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Sonuçları çizme ve mesafe hesaplama
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Avuç içi algılama: Elin altındaki noktaların yukarıda olması
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

            if index_mcp.y < wrist.y and pinky_mcp.y < wrist.y:
                # Avuç içi yukarıda, sesi kıs
                volume.SetMasterVolumeLevel(vol_min, None)
                cv2.putText(frame, "El algilandi - sistem calisiyor", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Avuç içi aşağıda, sesi eski seviyeye getir
                volume.SetMasterVolumeLevel(previous_volume, None)

            # Başparmak ve işaret parmağı için gerekli koordinatları al
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Piksel değerlerini hesapla
            h, w, _ = frame.shape
            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_tip_x, index_tip_y = int(index_tip.x * w), int(index_tip.y * h)

            # Mesafe hesaplama
            distance = math.hypot(index_tip_x - thumb_tip_x, index_tip_y - thumb_tip_y)

            # Mesafeyi 0 ile 100 arasında normalleştirme
            normalized_distance = max(0, min(100, ((distance - 30) / (200 - 30)) * 100))

            # Ses seviyesini 0'dan 100'e kadar ayarla
            volume_level = (normalized_distance / 100) * (vol_max - vol_min) + vol_min
            previous_volume = volume_level  # Eski ses seviyesini güncelle
            volume.SetMasterVolumeLevel(volume_level, None)

            # Mesafeyi ve ses seviyesini ekrana yazdırma
            cv2.putText(frame, f"Mesafe: {int(normalized_distance)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Seviye: {int((volume_level - vol_min) / (vol_max - vol_min) * 100)}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Mesafe noktasını çizme
            cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (index_tip_x, index_tip_y), 10, (0, 255, 0), -1)

    # Çıktıyı gösterme
    cv2.imshow('Hand and Face Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
