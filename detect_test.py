import pyaudio
import librosa
import numpy as np


class PitchDetect:
    # note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_names = ["도", "도#", "레", "레#", "미", "파", "파#", "솔", "솔#", "라", "라#", "시"]

    def get_stft(self, sample_sounds):
        return librosa.stft(sample_sounds)

    def get_chroma(self, sample_sounds, sr=44100):
        return librosa.feature.chroma_stft(
            S=np.abs(self.get_stft(sample_sounds)), sr=sr
        )

    def get_note_name_from_sound(self, sample_sounds, sr=44100):
        chroma_result = self.get_chroma(sample_sounds, sr=sr).mean(axis=1)
        note_index = np.argmax(np.abs(chroma_result))

        return f"{self.note_names[note_index]}"

    def get_note_name(self, frequency):
        if frequency < 100.0:
            return "empty"

        A4_freq = 440.0
        semitone_ratio = 2 ** (1 / 12.0)

        try:
            num_semitones = round(12 * np.log2(frequency / A4_freq))

        except:
            return "empty"
        octave = num_semitones // 12
        note_index = num_semitones % 12

        return f"{self.note_names[note_index]} {octave}"

    def analyze_audio_stream(self):
        CHUNK = 2**10
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 44100

        p = pyaudio.PyAudio()

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        print("Listening...")

        try:
            while True:
                data = stream.read(CHUNK)
                audio_array = np.frombuffer(data, dtype=np.float32)

                result = np.fft.fft(audio_array)
                # result = self.get_stft(audio_array).mean(axis=1)

                peak_freq_index = np.argmax(np.abs(result))
                frequency = (peak_freq_index * RATE) / CHUNK

                if 10000 > frequency > 50:
                    note_name = self.get_note_name_from_sound(audio_array)
                else:
                    note_name = "None"
                # note_name = self.get_note_name(frequency)
                # if frequency>=100:
                #     note_name += " "+self.get_note_name_from_sound(audio_array)

                print(
                    f"Detected frequency: {frequency:.2f} Hz | Note: {note_name}",
                )

        except KeyboardInterrupt:
            pass
        finally:
            print("\nStopped..")
            stream.stop_stream()
            stream.close()
            p.terminate()


if __name__ == "__main__":
    detector = PitchDetect()
    detector.analyze_audio_stream()
