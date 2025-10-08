import serial
import serial.tools.list_ports
import time

class ArduinoInterface:
    def __init__(self, port="COM5", baudrate=9600, timeout=1, inital_speed = 105):
        try:
            # Verbindung mit dem angegebenen Port herstellen
            self.serial_connection = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2)  # Wartezeit für die Arduino-Initialisierung
            print(f"Verbunden mit Arduino auf {port}")
            self.set_speed(inital_speed)
        except serial.SerialException as e:
            print(f"Fehler beim Verbinden mit dem Arduino: {e}")
            raise

    def set_speed(self, speed):
        if 0 <= speed <= 255:
            command = f"{speed}\n"
            self.serial_connection.write(command.encode())
            print(f"Gesendete Geschwindigkeit: {speed}")
        else:
            raise ValueError("Geschwindigkeit muss zwischen 0 und 100 liegen.")

    def close(self):
        self.serial_connection.close()
        print("Verbindung zum Arduino geschlossen.")
