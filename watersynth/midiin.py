"""MIDI vstup (volitelný): otevře všechny dostupné porty přes mido.

Když mido / python-rtmidi není nainstalované nebo žádné zařízení
nepřipojené, nástroj běží dál jen s počítačovou klávesnicí.
"""


class MidiInput:
    def __init__(self):
        self.ports = []
        self.status = "MIDI: nedostupné (pip install mido python-rtmidi)"
        try:
            import mido
        except ImportError:
            return
        try:
            names = mido.get_input_names()
            for name in names:
                self.ports.append(mido.open_input(name))
            self.status = (f"MIDI: {len(names)} vstup(y): " + ", ".join(names)
                           if names else "MIDI: žádné zařízení")
        except Exception as e:  # chybějící backend apod.
            self.status = f"MIDI: {e}"

    def poll(self):
        """Vrátí seznam událostí ('on', nota, velocity 0..1) / ('off', nota, 0)."""
        events = []
        for port in self.ports:
            for msg in port.iter_pending():
                if msg.type == "note_on" and msg.velocity > 0:
                    events.append(("on", msg.note, msg.velocity / 127.0))
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    events.append(("off", msg.note, 0.0))
        return events

    def close(self):
        for port in self.ports:
            port.close()
