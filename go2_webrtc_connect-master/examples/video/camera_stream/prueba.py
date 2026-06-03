import cv2
import numpy as np
import asyncio
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
# --- CONFIGURACIÓN ---
# ID de clase para "persona" en MobileNet-SSD es 15
PERSON_CLASS_ID = 15
CONFIDENCE_THRESHOLD = 0.5
TARGET_DISTANCE = 1.2  # Distancia ideal en metros
DISTANCE_TOLERANCE = 0.2

# Cargar modelo SSD
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

class AutoFollower:
    def __init__(self):
        self.conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)
        self.last_lidar_dist = 0.0

    async def start(self):
        await self.conn.connect()
        print("Conectado al Go2 vía AP Mode")
        
        # Listener para el Lidar (asumimos que el driver nos da la distancia frontal)
        self.conn.set_lidar_callback(self.on_lidar_data)
        
        # Procesar Video
        while True:
            frame = self.conn.get_video_frame()
            if frame is not None:
                self.process_frame(frame)
            await asyncio.sleep(0.03) # ~30 FPS

    def on_lidar_data(self, cloud):
        # Simplificación: obtenemos el punto central más cercano del Lidar
        # El driver de legion1581 decodifica PointCloud; aquí promediamos el centro
        self.last_lidar_dist = np.mean(cloud[len(cloud)//2]) 

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        found_person = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                class_id = int(detections[0, 0, i, 1])
                
                if class_id == PERSON_CLASS_ID:
                    # Coordenadas de la caja
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Calcular centro de la persona
                    center_x = (startX + endX) / 2
                    error_x = (center_x - (w / 2)) / (w / 2) # Normalizado -1 a 1
                    
                    self.follow_logic(error_x)
                    found_person = True
                    break # Seguimos solo a la primera persona detectada

        if not found_person:
            self.stop_robot()

    def follow_logic(self, error_x):
        # Velocidad angular (Giro)
        yaw_speed = -error_x * 0.5 # Sensibilidad de giro
        
        # Velocidad lineal (Avanzar/Retroceder según Lidar)
        forward_speed = 0.0
        dist_error = self.last_lidar_dist - TARGET_DISTANCE
        
        if abs(dist_error) > DISTANCE_TOLERANCE:
            forward_speed = np.clip(dist_error * 0.4, -0.3, 0.4)

        # Enviar comando al Go2 (High-level move)
        # x: adelante/atrás, y: lateral, z: rotación
        self.conn.move(x=forward_speed, y=0, z=yaw_speed)

    def stop_robot(self):
        self.conn.move(0, 0, 0)

if __name__ == "__main__":
    follower = AutoFollower()
    asyncio.run(follower.start())