import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from Arduino import ArduinoInterface
from Camera import RealSenseCamera
import time
import timeit
import statistics 

class NFQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NFQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ----------------
# CarreraTrackEnv
# ----------------
class CarreraTrackEnv:
    def __init__(self, positions, arduino):
        """
        Initialisiert die Umgebung mit den Referenzpunkten und der Arduino-Schnittstelle.
        :param positions: Liste der Referenzpunkte (x, y) auf der Strecke.
        :param arduino: Objekt der ArduinoInterface-Klasse zur Steuerung der Geschwindigkeit.
        """
        self.positions = positions
        self.arduino = arduino

        self.recent_positions = []
        self.recent_times = []
        self.smoothed_speed = 0
        self.max_batch_size = 5   # Batch-Größe für Geschwindigkeitsberechnung
        self.done = False
        self.current_index = 0
        self.off_track = False
        self.last_off_track = False  # Zustand für Off-Track Status
        self.step_count = 0
        self.max_steps = 200  # Maximale Anzahl an Schritten pro Episode

    def reset(self):
        """
        Setzt die Umgebung zurück und gibt den initialen Zustand zurück.
        """
        self.recent_positions.clear()
        self.recent_times.clear()
        self.smoothed_speed = 0
        self.done = False
        self.current_index = 0
        self.off_track = False
        self.last_off_track = False
        self.step_count = 0

        # Zustand: [geschätzte Geschwindigkeit, Fortschritt, off_track?]
        # Hier z.B. geschätzte Geschwindigkeit = 0, progress = 0, off_track = 0
        return np.array([0.0, 0.0, 0], dtype=np.float32)

    def step(self, position, progress,speed, off_track):
        """
        Führt einen Schritt in der Umgebung aus, basierend auf den RealSense-Daten.
        Ruft intern Methoden der RealSenseCamera-Klasse auf, um Position, Off-Track,
        Geschwindigkeit etc. zu bestimmen.

        :return: (next_state, progress, off_track, done, reward)
        """
        self.step_count += 1

        # 1) Kamera-Daten holen
        current_position = position   # z.B. (x, y)

        # 2) Off-Track-Check
        self.off_track = off_track

        # 3) Geschätzte Geschwindigkeit ermitteln (z.B. pixel/s)
        speed = speed

        # 4) Fortschritt berechnen
        #    Beispiel: Du hast evtl. eine Methode camera.get_next_point_and_coverage(position),
        #    die sowohl den "nächsten Punkt" als auch "Coverage %" zurückgibt.
        #    Dann könnte man so was machen:
    
        # coverage ist in Prozent (0..100)
        progress = progress

        # 5) Fertig-Bedingung prüfen (z.B. coverage >= 98% oder max_steps überschritten)
        if progress >= 98 or self.step_count >= self.max_steps:
            self.done = True

        # 6) Belohnung
        if self.off_track:
            reward = -100.0
        else:
            # Einfaches Reward-Schema: je höher speed, desto besser
            # (z.B. speed / 10.0 oder speed / 5.0)
            reward = speed / 10.0

        # 7) Nächster Zustand
        # off_track als 0 oder 1
        off_track_flag = 1 if self.off_track else 0
        next_state = np.array([speed, progress, off_track_flag], dtype=np.float32)
        print(f"Step {self.step_count}: Reward={reward:.2f}, Speed={speed:.2f}, Progress={progress:.2f}, Off-Track={self.off_track}")  
        return next_state, progress, self.off_track, self.done, reward

    def get_current_time(self):
        """
        Gibt die aktuelle Zeit in Sekunden zurück.
        """
        return time.time()


# ----------------
# NFQAgent
# ----------------
class NFQAgent:
    def __init__(self, arduino):
        # Zwei Q-Netze (Policy + Target)
        self.policy_network = NFQNetwork(input_dim=3, output_dim=256)
        self.target_network = NFQNetwork(input_dim=3, output_dim=256)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.arduino = arduino
        self.target_update_frequency = 10
        self.step_count = 0

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        """
        Wählt eine Aktion (Geschwindigkeit 0..255).
        """
        if random.uniform(0, 1) < self.epsilon:  # Exploration
            action = random.randint(0, 255)
        else:  # Exploitation
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_network(state_t)  # [1 x 256]
                action = q_values.argmax(dim=1).item()

        # An Arduino senden
        self.arduino.set_speed(action)
        return action

    def replay(self):
        """
        Replay mit Target und Policy Network.
        """
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        batch_data = list(zip(*batch))

        states = torch.tensor(np.array(batch_data[0]), dtype=torch.float32)
        actions = torch.tensor(batch_data[1], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(batch_data[2], dtype=torch.float32)
        next_states = torch.tensor(np.array(batch_data[3]), dtype=torch.float32)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q_values

        current_q_values = self.policy_network(states).gather(1, actions).squeeze()

        loss = self.criterion(current_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon-Decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        # Target-Net aktualisieren
        if self.step_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
