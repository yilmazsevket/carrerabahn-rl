from Camera import RealSenseCamera
from Arduino import ArduinoInterface
from CarreraTrackEnvQ import CarreraTrackEnv,NFQAgent

def main():
    # A) Arduino starten
    arduino = ArduinoInterface(port="COM5", baudrate=9600, timeout=1, inital_speed=105)

    # B) RealSense-Kamera init
    camera = RealSenseCamera()

    # 1) Starpunkt setzen
    input("Bitte Auto am Startpunkt platzieren und Enter drücken...")
    camera.set_start_point()
    # 2) Runde aufzeichnen
    recorded_positions = camera.record_round(
        min_distance_left=50,
        tolerance_return=30
    )


    # Falls nichts aufgezeichnet => Notausstieg
    if len(camera.positions) < 2:
        print("Keine sinnvollen Streckenpunkte. Programmende.")
        arduino.set_speed(0)
        arduino.close()
        return

    # D) Umgebung mit den aufgezeichneten Positionen erstellen
    env = CarreraTrackEnv(positions=recorded_positions, arduino=arduino)

    # E) Agent erstellen
    agent = NFQAgent(arduino=arduino)

    # F) Trainingsloop
    num_episodes = 100
    max_steps = 200

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode+1} ===")
        state = env.reset()
        done = False


        for step in range(max_steps):
            # Aktion wählen
            action = agent.act(state)

            # Hier echte Position via RealSense holen 
            # (im Beispiel simulieren wir es, 
            #  indem wir zufällig aus recorded_positions wählen oder so)
            current_position = camera.get_position()
            speed= camera.get_average_velocity()
            off_track = camera.is_offtrack(current_position)
            next_pt, progress = camera.get_next_point_and_coverage(current_position)
            # Schritt
            next_state, progress, off_track, done, reward = env.step(current_position, progress,speed, off_track)

            # Lernen
            agent.remember(state, action, reward, next_state)
            agent.replay()

            state = next_state

            if done:
                print(f"Episode {episode+1}: nach {step+1} Schritten DONE. Reward={reward:.2f}")
                break
        else:
            print(f"Episode {episode+1}: max. Steps ({max_steps}) erreicht.")

    # G) Nach dem Training alles schließen
    arduino.set_speed(0)
    arduino.close()
    print("Training beendet.")


if __name__ == "__main__":
    main()