import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random


class CarreraTrackEnv(gym.Env):
    def __init__(self, track_positions, arduino):
        super(CarreraTrackEnv, self).__init__()
        
        # Die Liste von Track-Punkten (aufgenommen durch Initialfahrt)
        self.track_positions = track_positions  # Liste von (x, y)-Punkten
        self.num_points = len(self.track_positions)

        # Arduino-Schnittstelle zur Geschwindigkeitssteuerung
        self.arduino = arduino
        
        # Observation Space: [Position in % der Strecke, Geschwindigkeit (Punkte/10 Frames), Off-Track-Status]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 70, 0]),  # min: 0% Fortschritt, 0 Geschwindigkeit, nicht Off-Track
            high=np.array([100.0, 255, 1]),  # max: 100% Fortschritt, hypothetische maxspeed (maximale Spannung). Geschwindigkeit, Off-Track
            dtype=np.float32
        )
        
        # Action Space: Geschwindigkeit (70 bis 255 kontinuierlich)
        self.action_space = spaces.Box(low=70, high=255, shape=(1,), dtype=np.float32)
        
        # Interne Variablen
        self.current_position_index = 0
        self.progress = 0.0  # Fortschritt in %
        self.speed = 0.0  # Punkte pro 10 Frames
        self.off_track = False
        self.previous_position = None
        self.steps = 0

        # Q-Learning-Parameter
        self.epsilon = 1.0  # Erkundungsrate (Wahrscheinlichkeit für zufällige Aktion)
        self.epsilon_decay = 0.995  # Rate, mit der die Erkundung abnimmt
        self.epsilon_min = 0.01  # Minimalwert für die Erkundungsrate
        self.alpha = 0.1  # Lernrate (wie stark neue Informationen alte überschreiben)
        self.gamma = 0.99  # Discountfaktor (Wichtigkeit zukünftiger Belohnungen)
        self.q_table = np.zeros((101, 226, 2, 186))  # Diskretisierung des Zustandsraums zur Vereinfachung
        self.frame_batch = []  # Neu: Liste zur Speicherung der letzten 5 Frames

    def step(self, real_position):
        """
        Simuliert einen Schritt der Umgebung basierend auf 5 Frames.
        :param real_position: Die aktuelle Position des Autos auf der Strecke
        """
        self.frame_batch.append(real_position)

        # Überprüfe jeden Frame auf "Off-Track"
        for position in self.frame_batch:
            distance_to_track_points = [
                np.linalg.norm(np.array(position) - np.array(track_point))
                for track_point in self.track_positions
            ]
            nearest_index = int(np.argmin(distance_to_track_points))
            if distance_to_track_points[nearest_index] > 10:  # Toleranzradius
                self.off_track = True
                break
        else:
            self.off_track = False

        # Nur alle 5 Frames die Fortschritt- und Geschwindigkeitsberechnungen durchführen
        if len(self.frame_batch) < 5:
            return self._get_observation(), 0, False, {}

        # Durchschnittliche Position über die letzten 5 Frames berechnen
        avg_position = np.mean(self.frame_batch, axis=0).astype(int)
        self.frame_batch = []  # Batch zurücksetzen

        # Berechne die Distanz zur aktuellen Position auf der Strecke
        distance_to_track_points = [
            np.linalg.norm(np.array(avg_position) - np.array(track_point))
            for track_point in self.track_positions
        ]
        nearest_index = int(np.argmin(distance_to_track_points))

        # Fortschritt in %
        self.progress = (nearest_index / (self.num_points - 1)) * 100

        # Geschwindigkeit (Punkte/5 Frames)
        distance_moved = (nearest_index - self.current_position_index) % self.num_points
        self.speed = distance_moved
        self.current_position_index = nearest_index

        # Belohnungslogik
        reward = -10 if self.off_track else 1 + self.speed / 10

        # Ziel erreicht?
        if self.progress >= 100:
            reward += 100
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def _choose_action(self, state):
        """
        Wählt eine Aktion basierend auf der ε-greedy Strategie.
        """
        if random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()  # Zufällige Aktion (Geschwindigkeit)
            action = np.clip(action, 70, 255)  # Sicherstellen, dass Aktion im zulässigen Bereich liegt
            print(f"Random Action Chosen: {action[0]}")
            return action[0]
        else:
            action = np.argmax(self.q_table[state[0], state[1], state[2], :]) + 70  # Beste bekannte Aktion (Geschwindigkeit)
            print(f"Best Action Chosen: {action}")
            return action

    def _get_observation(self):
        """
        Liefert die aktuelle Beobachtung zurück.
        """
        return np.array([self.progress, self.speed, int(self.off_track)], dtype=np.float32)

    def reset(self):
        """
        Setzt die Umgebung zurück.
        """
        self.progress = 0.0
        self.speed = 0.0
        self.off_track = False
        self.current_position_index = 0
        self.previous_position = self.track_positions[0]
        self.done = False
        
        # Reduziere die Erkundungsrate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Debug-Ausgabe für Reset
        print(f"Environment Reset: Epsilon: {self.epsilon:.4f}")
        
        # Gib den initialen Zustand und ein leeres Wörterbuch zurück
        return self._get_observation(), {}

    def _set_speed(self, action):
        """
        Setzt die Geschwindigkeit basierend auf der gewählten Aktion.
        """
        speed_value = action  # Aktion ist jetzt eine kontinuierliche Geschwindigkeit zwischen 70 und 255
        self.arduino.set_speed(speed_value)
        print(f"Setting speed to: {speed_value}")

    def render(self, mode="human"):
        # Das gleiche Render-Code wie vorher

        track_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        max_x = max([point[0] for point in self.track_positions])
        max_y = max([point[1] for point in self.track_positions])
        scale_x = 1280 / (max_x + 50)
        scale_y = 720 / (max_y + 50)

        for point in self.track_positions:
            x, y = int(point[0] * scale_x), int(point[1] * scale_y)
            track_image = cv2.circle(track_image, (x, y), 6, (255, 255, 255), -1)

        if not self.off_track:
            current_point = self.track_positions[self.current_position_index]
            x, y = int(current_point[0] * scale_x), int(current_point[1] * scale_y)
            track_image = cv2.circle(track_image, (x, y), 10, (0, 255, 0), -1)
        else:
            track_image = cv2.putText(track_image, "Off Track", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        track_image = cv2.putText(track_image, f"Speed: {self.speed:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        track_image = cv2.putText(track_image, f"Progress: {self.progress:.2f}%", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Carrera Track", track_image)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()