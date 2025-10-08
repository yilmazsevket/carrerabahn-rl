#include <SPI.h>

const int CS_PIN = 10;   // Chip-Select-Pin für den MCP41X1
int previousValue = -1;  // Speichert den letzten eingestellten Wert

void setup() {
  Serial.begin(9600);       // Startet die serielle Kommunikation
  SPI.begin();              // Initialisiert SPI
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);  // CS auf HIGH, wenn nicht aktiv
  Serial.println("Geschwindigkeitssteuerung bereit.");
}

void loop() {
  if (Serial.available() > 0) {                  
    String input = Serial.readStringUntil('\n');  
    int rawValue = validateInput(input.toInt());

    // Skalierung von 0-100 auf 60-255
    int scaledValue = map(rawValue, 0, 100, 60, 255);
    if (scaledValue != previousValue) {      // Prüft, ob der neue Wert sich vom alten unterscheidet
      setPotentiometer(scaledValue);         // Setzt direkt ohne Verzögerung
      previousValue = scaledValue;           // Aktualisiert den letzten Wert
      sendFeedbackToRL(previousValue, rawValue);
    }
  }
}

// Funktion zur Validierung der Eingabe (0 bis 100)
int validateInput(int value) {
  if (value < 0) return 0;
  if (value > 100) return 100;
  return value;
}

// Funktion zum Einstellen des Potentiometerwerts über SPI
void setPotentiometer(byte value) {
  SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE0));  // Setzt SPI Geschwindigkeit auf 2 MHz
  digitalWrite(CS_PIN, LOW);             // CS aktivieren
  SPI.transfer(0);                       // Steuerbefehl senden
  SPI.transfer(value);                   // Übermittelt den gewünschten Widerstandswert
  digitalWrite(CS_PIN, HIGH);            // CS deaktivieren
  SPI.endTransaction();
  
  // Ausgabe zur Kontrolle
  Serial.print("Potentiometerwert eingestellt auf: ");
  Serial.println(value);
}

// Funktion zur Rückmeldung für das RL-Modell
void sendFeedbackToRL(int currentSpeed, int targetSpeed) {
  Serial.print("Aktuelle Geschwindigkeit (Potentiometer): ");
  Serial.print(currentSpeed);
  Serial.print(" | Zielgeschwindigkeit (0-100): ");
  Serial.println(targetSpeed);
}