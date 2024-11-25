import socket
import cv2
import numpy as np
import threading
import struct
import time
import mediapipe as mp
import tensorflow as tf

class VideoStreamServer:
    def __init__(self, ip='0.0.0.0', port=9999, buffer_size=4096, timeout=5):
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.timeout = timeout  # Timeout in seconds
        self.clients = {}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        print(f"Server listening on {self.ip}:{self.port}")

        # Initialize a thread to monitor inactive clients
        self.monitor_thread = threading.Thread(target=self.monitor_clients, daemon=True)
        self.monitor_thread.start()

        # Initialize Mediapipe and TensorFlow Lite models
        self.mp_holistic = mp.solutions.holistic  # Holistic model
        self.mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

        # Load TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path="model.tflite")
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.sequence_length = self.input_details[0]['shape'][1]  # Number of timesteps
        self.expected_features = self.input_details[0]['shape'][2]  # Features per timestep

        # Define actions and variables
        self.actions = np.array(['Ăn uống', 'Cảm ơn', 'Lặp lại', 'Mẹ', 'Nguy hiểm', 
                                 'Ngủ', 'Tắm', 'Tạm biệt', 'Thêm', 'Xin chào', 'Xin lỗi'])
        self.sequence = []
        self.sentence = []
        self.threshold = 0.8

    def start(self):
        try:
            with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while True:
                    data, addr = self.sock.recvfrom(self.buffer_size)
                    current_time = time.time()

                    if addr not in self.clients:
                        print(f"New client connected: {addr}")
                        self.clients[addr] = {
                            'frames': {},
                            'expected_packets': {},
                            'lock': threading.Lock(),
                            'last_active': current_time
                        }
                        # Send a welcome message to the client
                        welcome_message = "Sẵn Sàng Nhận Diện...".encode('utf-8')
                        self.sock.sendto(welcome_message, addr)
                        print(f"Sent welcome message to client {addr}")

                    # Update client's last active time
                    with self.clients[addr]['lock']:
                        self.clients[addr]['last_active'] = current_time

                    # Decode header
                    if len(data) >= 8:
                        frame_id, packet_id = struct.unpack('!II', data[:8])
                        if packet_id == 0xFFFFFFFF:  # Check for footer
                            if len(data) >= 12:
                                total_packets = struct.unpack('!I', data[8:12])[0]
                                with self.clients[addr]['lock']:
                                    self.clients[addr]['expected_packets'][frame_id] = total_packets
                                print(f"Received footer for frame {frame_id} from {addr}, total_packets: {total_packets}")

                                # Check if all packets for a frame have been received
                                with self.clients[addr]['lock']:
                                    if frame_id in self.clients[addr]['frames']:
                                        received_packets = len(self.clients[addr]['frames'][frame_id])
                                        if received_packets == total_packets:
                                            print(f"All packets received for frame {frame_id} from {addr}")
                                            # Merge data
                                            frame_data = b''.join([self.clients[addr]['frames'][frame_id][i] for i in range(total_packets)])
                                            self.handle_frame(addr, frame_data, frame_id, holistic)
                                            # Delete processed data
                                            del self.clients[addr]['frames'][frame_id]
                                            del self.clients[addr]['expected_packets'][frame_id]
                                    else:
                                        print(f"Frame {frame_id} chưa nhận được tất cả các gói tin. Đợi thêm dữ liệu...")
                        else:
                            chunk = data[8:]
                            with self.clients[addr]['lock']:
                                if frame_id not in self.clients[addr]['frames']:
                                    self.clients[addr]['frames'][frame_id] = {}
                                self.clients[addr]['frames'][frame_id][packet_id] = chunk
                            print(f"Received packet {packet_id} for frame {frame_id} from {addr}")

                            # Check if all packets for a frame have been received
                            with self.clients[addr]['lock']:
                                if frame_id in self.clients[addr]['expected_packets']:
                                    total_packets = self.clients[addr]['expected_packets'][frame_id]
                                    received_packets = len(self.clients[addr]['frames'][frame_id])
                                    if received_packets == total_packets:
                                        print(f"All packets received for frame {frame_id} from {addr}")
                                        # Merge data
                                        frame_data = b''.join([self.clients[addr]['frames'][frame_id][i] for i in range(total_packets)])
                                        self.handle_frame(addr, frame_data, frame_id, holistic)
                                        # Delete processed data
                                        del self.clients[addr]['frames'][frame_id]
                                        del self.clients[addr]['expected_packets'][frame_id]

        except KeyboardInterrupt:
            print("\nServer đang tắt...")
        finally:
            self.sock.close()

    def handle_frame(self, addr, data, frame_id, holistic):
        frame = self.decode_frame(data)
        if frame is not None:
            # Process frame for sign language recognition
            image, results = self.mediapipe_detection(frame, holistic)
            self.draw_landmarks(image, results)

            # Extract keypoints
            keypoints = self.extract_keypoints(results)
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-self.sequence_length:]  # Keep the last N timesteps

            # Perform prediction if sequence is ready
            if len(self.sequence) == self.sequence_length:
                input_data = np.expand_dims(self.sequence, axis=0).astype(self.input_details[0]['dtype'])  # Shape: [1, timesteps, features]
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

                # Determine action
                if output_data[np.argmax(output_data)] > self.threshold:
                    if len(self.sentence) == 0 or self.actions[np.argmax(output_data)] != self.sentence[-1]:
                        self.sentence.append(self.actions[np.argmax(output_data)])

                # Limit sentence to the last action
                self.sentence = self.sentence[-1:]
            
            # In kết quả nhận diện ra console
            recognition_result = ' '.join(self.sentence)
            print(f"Client {addr}: {recognition_result}")
            # Gửi kết quả nhận diện về cho client
            try:
                response_data = recognition_result.encode('utf-8')
                self.sock.sendto(response_data, addr)
                print(f"Gửi kết quả nhận diện về cho client {addr}")
            except Exception as e:
                print(f"Lỗi khi gửi dữ liệu về client {addr}: {e}")

            # Display the frame with landmarks
            cv2.imshow(f"Recognition - Client {addr}", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

    def decode_frame(self, data):
        np_data = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return frame

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image.flags.writeable = False  # Optimize for inference
        results = model.process(image)  # Make predictions
        image.flags.writeable = True  # Make image writable again
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
        return image, results

    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)  # Left hand
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)  # Right hand

    def extract_keypoints(self, results):
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([lh, rh])  # Total: 126 features

    def monitor_clients(self):
        """
        This thread continuously checks for inactive clients. If a client doesn't send data within the timeout period, it disconnects the client.
        """
        while True:
            current_time = time.time()
            inactive_clients = []

            # Find inactive clients
            for addr, info in list(self.clients.items()):
                with info['lock']:
                    if current_time - info['last_active'] > self.timeout:
                        inactive_clients.append(addr)

            # Disconnect inactive clients
            for addr in inactive_clients:
                with self.clients[addr]['lock']:
                    print(f"Client {addr} đã không hoạt động trong {self.timeout} giây. Ngắt kết nối.")
                    del self.clients[addr]

            time.sleep(1)  # Check every second

    def __del__(self):
        self.sock.close()

if __name__ == "__main__":
    server = VideoStreamServer()
    server.start()