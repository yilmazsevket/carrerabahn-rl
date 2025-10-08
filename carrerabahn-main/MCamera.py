import cv2
import numpy as np
import pyrealsense2 as rs
import time
import math
import matplotlib.pyplot as plt
from collections import deque

def distance(p1, p2):
    """Euklidische Distanz zweier Punkte (x1, y1) zu (x2, y2)."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
        self.align = rs.align(rs.stream.color)  # Align depth to color stream

        try:
            self.pipeline.start(self.config)
        except Exception as e:
            print(f"Fehler beim Starten der RealSense-Kamera: {e}")
            exit(1)

        self.positions = []               # Gesamter Pfad
        self.start_point = None
        self.last_position = None

        # Letzte 5 Einträge als (timestamp, x, y) für Geschwindigkeiten:
        self.last_five_positions = deque(maxlen=5)

        # Parameter für "Runde-Aufzeichnen"
        self.tolerance_return = 20
        self.min_distance_to_count_as_left = 30

    def stop(self):
        """Beendet den Kamerastream."""
        self.pipeline.stop()
        print("Kamera gestoppt")

    def smooth_position(self, new_position, alpha=0.5):
        """
        Einfache Glättungsfunktion (Exponential Moving Average).
        """
        if self.last_position is None:
            return new_position
        x_new = int(alpha * new_position[0] + (1 - alpha) * self.last_position[0])
        y_new = int(alpha * new_position[1] + (1 - alpha) * self.last_position[1])
        return (x_new, y_new)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            raise RuntimeError("Frames nicht verfügbar")

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image

    def get_position(self):
        """
        Ermittelt die aktuelle Position eines grünen Objekts (falls vorhanden).
        Gibt (x, y) zurück oder None, falls kein Objekt gefunden wird.
        """
        color_image, _ = self.get_frame()
        hsv_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 130
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]

        if contours and cv2.countNonZero(mask) > 20:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Glätten
                pos = self.smooth_position((cX, cY))
                self.last_position = pos

                # Zeitstempel + x + y
                current_time = time.time()
                self.last_five_positions.append((current_time, pos[0], pos[1]))

                return pos
        return None

    def set_start_point(self):
        """Versuche einmalig, einen Startpunkt zu finden."""
        for _ in range(50):
            pos = self.get_position()
            if pos is not None:
                self.start_point = pos
                self.positions.append(pos)
                print(f"Startpunkt gesetzt: {pos}")
                return
        print("Kein Startpunkt gefunden!")

    def record_round(self):
        """
        Zeichnet die Runde auf, bis die Vollrunde erkannt wird.
         1) Warte, bis die Distanz > min_distance_to_count_as_left
         2) Beende, wenn wir wieder in tolerance_return Nähe vom Start sind
        """
        if self.start_point is None:
            print("Kein Startpunkt definiert. Breche ab.")
            return

        has_left_start = False
        print("Aufzeichnung läuft. Sobald eine Vollrunde erkannt wird, endet das Programm...")

        while True:
            current_pos = self.get_position()
            if current_pos:
                self.positions.append(current_pos)
                
                # Prüfen, ob wir den Startpunkt bereits verlassen haben
                if not has_left_start:
                    if distance(current_pos, self.start_point) > self.min_distance_to_count_as_left:
                        has_left_start = True
                        print("Der Startpunkt wurde jetzt wirklich verlassen.")
                else:
                    # Prüfen, ob wir wieder nahe am Startpunkt sind => Vollrunde
                    if distance(current_pos, self.start_point) < self.tolerance_return:
                        print("Vollrunde erkannt: wieder am Startpunkt angekommen.")
                        break

    def interpolate_positions(self, positions, target_distance=10.0):
        """
        Interpoliert Punkte entlang der Strecke, um eine gleichmäßige Verteilung zu erreichen.
        """
        if len(positions) < 2:
            return positions

        interpolated_positions = [positions[0]]
        for i in range(1, len(positions)):
            x1, y1 = positions[i - 1]
            x2, y2 = positions[i]
            dist_pts = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist_pts == 0:
                interpolated_positions.append((x2, y2))
                continue

            num_new_points = max(1, int(dist_pts // target_distance))
            for t in np.linspace(0, 1, num_new_points + 1)[1:]:
                x = (1 - t) * x1 + t * x2
                y = (1 - t) * y1 + t * y2
                interpolated_positions.append((x, y))

        return interpolated_positions

    def plot_positions(self, positions=None, title="Aufgezeichnete Strecke"):
        """
        Zeichnet die übergebenen (x, y)-Positionen mit matplotlib.
        """
        if positions is None:
            positions = self.positions
        
        if not positions:
            print("Keine Positionen zum Plotten vorhanden!")
            return

        all_x = [p[0] for p in positions]
        all_y = [p[1] for p in positions]

        plt.figure(figsize=(6, 6))
        plt.plot(all_x, all_y, marker='o', linestyle='-', color='g', label='Bewegung')
        plt.scatter(all_x[0], all_y[0], color='r', s=60, label='Startpunkt')
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()

    def is_offtrack(self, car_position):
        """
        Prüft, ob das Auto (car_position) noch auf der Strecke ist.
        Kriterium:
         - max. Abstand zwischen je zwei benachbarten Punkten in self.positions
         - ist distance(car_position, nearest_point) > maxNachbarAbstand => offtrack
        """
        if not self.positions or len(self.positions) < 2:
            return False

        # 1) Maximalen Nachbarabstand bestimmen
        max_neigh_dist = 0
        for i in range(1, len(self.positions)):
            d_pts = distance(self.positions[i], self.positions[i - 1])
            if d_pts > max_neigh_dist:
                max_neigh_dist = d_pts

        # 2) Nächsten Punkt zur car_position in self.positions finden
        nearest_dist = min(distance(car_position, p) for p in self.positions)

        # 3) off-track?
        return nearest_dist > max_neigh_dist

    def get_average_velocity(self):
        """
        Berechnet die durchschnittliche Geschwindigkeit in "Pixel pro Sekunde"
        basierend auf den letzten 5 Einträgen (time, x, y).
        """
        if len(self.last_five_positions) < 2:
            return 0.0

        last_positions = list(self.last_five_positions)
        total_dist = 0.0
        
        # Summiere die Abstände zwischen aufeinanderfolgenden Positionen
        for i in range(1, len(last_positions)):
            t0, x0, y0 = last_positions[i - 1]
            t1, x1, y1 = last_positions[i]
            total_dist += distance((x0, y0), (x1, y1))

        # Berechne die verstrichene Zeit
        t_start = last_positions[0][0]
        t_end = last_positions[-1][0]
        delta_t = t_end - t_start

        if delta_t <= 0:
            return 0.0

        avg_velocity = total_dist / delta_t
        return avg_velocity

    def get_last_5_coverage_ratio(self):
        """
        Berechnet:
         1) die direkte Distanz zwischen dem ERSTEN und dem LETZTEN
            Eintrag in self.last_five_positions (=> "in den letzten 5 Frames")
         2) den prozentualen Anteil dieser Distanz an der gesamten Streckenlänge (self.positions).

        Rückgabe: Prozentualer Wert (0..100), wie viel % der Gesamtstrecke
                  diese 5-Punkt-Distanz ausmacht.
        """
        # (A) Prüfen, ob wir mindestens 2 Einträge in last_five_positions haben
        if len(self.last_five_positions) < 2:
            return 0.0

        # Aus den Einträgen (t, x, y) nur (x, y) herausziehen (Index 1,2)
        last_positions_list = list(self.last_five_positions)
        x1, y1 = last_positions_list[0][1], last_positions_list[0][2]
        x5, y5 = last_positions_list[-1][1], last_positions_list[-1][2]

        dist_last_5 = distance((x1, y1), (x5, y5))

        # (B) Gesamtstrecke berechnen (summe der Abstände in self.positions)
        if len(self.positions) < 2:
            return 0.0

        total_dist = 0.0
        for i in range(1, len(self.positions)):
            total_dist += distance(self.positions[i - 1], self.positions[i])

        if total_dist == 0:
            return 0.0

        # (C) Prozentualer Anteil
        coverage_percent = (dist_last_5 / total_dist) * 100.0
        return coverage_percent


# ----------------------------------------------------------------------------
# Beispielhafte Nutzung
if __name__ == "__main__":
    camera = RealSenseCamera()

    # 1) Startpunkt festlegen
    input("Drücke Enter, um den Startpunkt zu setzen...")
    camera.set_start_point()

    # 2) Runde aufzeichnen
    camera.record_round()

    # 3) Kamera stoppen
    camera.stop()

    # Optional: Interpolation, Plot etc...
    # camera.positions = camera.interpolate_positions(camera.positions)
    # camera.plot_positions(ti
    #tle="Originale Positionen")

    # Beispiel: Wieviel % der Gesamtstrecke
    #   liegt direkt zwischen dem 1. und dem 5. der letzten Positions?
    ratio = camera.get_last_5_coverage_ratio()
    print(f"Letzte 5-Punkte-Distanz entspricht {ratio:.2f}% der gesamten Strecke.")
