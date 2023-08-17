import pyaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left


class PitchDetect:
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_names_kr = ["도", "도#", "레", "레#", "미", "파", "파#", "솔", "솔#", "라", "라#", "시"]
    octaves = [31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383]

    def __init__(self):
        self.CHUNK = 2**10
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100

    def get_frequencies(self, sample_sounds):
        fft_result = np.fft.fft(sample_sounds)

        peak_freq_index = np.argmax(np.abs(fft_result))
        frequency = (peak_freq_index * self.RATE) / self.CHUNK

        return frequency

    def get_stft(self, sample_sounds):
        return librosa.stft(sample_sounds)

    def get_chroma(self, sample_sounds, sr=0):
        return librosa.feature.chroma_stft(
            S=np.abs(self.get_stft(sample_sounds)), sr=sr
        )

    def get_note_name_from_sound(self, sample_sounds, frequency, sr=44100):
        # chroma_result = self.get_chroma(sample_sounds, sr=sr).mean(axis=1)
        # note_index = np.argmax(np.abs(chroma_result))
        octave = bisect_left(self.octaves, frequency)
        cur = 440 / (2 ** (4 - octave))
        note_index = 9

        while True:
            if frequency >= cur:
                next_ = cur * (2 ** (1 / 12))
                if frequency < next_:
                    if next_ - frequency < frequency - cur:
                        note_index += 1
                    break
                else:
                    cur = next_
                    note_index += 1
                    continue
            else:
                next_ = cur * (2 ** (-1 / 12))
                if frequency > next_:
                    if frequency - next_ < cur - frequency:
                        note_index -= 1
                    break
                else:
                    cur = next_
                    note_index -= 1
                    continue

        return (
            f"{self.note_names_kr[note_index]} ({self.note_names[note_index]}{octave})"
        )

    def analyze_audio_stream(self):
        p = pyaudio.PyAudio()

        stream = p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        print("Listening...")
        plt.ion()
        frequency_list = [0 for i in range(256)]
        frequency_index = 0
        fig = plt.figure()
        ax = plt.axes(xlim=(0, 256), ylim=(0, 1000))

        try:
            while True:
                data = stream.read(self.CHUNK)
                audio_array = np.frombuffer(data, dtype=np.int16)
                frequency = self.get_frequencies(audio_array)
                if 10000 > frequency > 50:
                    note_name = self.get_note_name_from_sound(
                        audio_array,
                        frequency=frequency,
                        sr=self.RATE,
                    )
                else:
                    frequency = 0
                    note_name = "None"
                    continue
                print(
                    f"Detected frequency: {frequency:.2f} Hz | Note: {note_name}",
                )

                frequency_list[frequency_index] = frequency
                frequency_index = (frequency_index + 1) % 256
                ax.clear()
                ax = plt.axes(xlim=(0, 256), ylim=(0, 1000))
                ax.plot(
                    frequency_list[frequency_index:] + frequency_list[:frequency_index]
                )
                plt.pause(0.01)

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
