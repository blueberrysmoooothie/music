import pyaudio
import numpy as np


RATE = 44100
CHUNK = int(RATE / 10)
LEN = 10

def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

    for i in range(500):
        data = np.fromstring(stream.read(CHUNK), dtype = np.int16)

        peak = np.average(np.abs(data))*2
        bars = '#'*int(50*peak/2**16)
        print("%04d %05d %s"%(i, peak, bars))
        
    
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__=='__main__':
    main()
    print('end main')

# p = pyaudio.PyAudio()
# for index in range(p.get_device_count()):
#     desc = p.get_device_info_by_index(index)
#     print("DEVICE: {device}, INDEX: {index}, RATE: {rate} ".format(
#         device=desc["name"], index=index, rate=int(desc["defaultSampleRate"])))






# for index in range(audio.get)

# stream = p.open(format=pyaudio.paInt16, channels=1, rate= RATE, input = True,
#                 frames_per_buffer=CHUNK, input_device_index=2)

# while True :
#     data = np.fromstring(stream.read(CHUNK), dtype = np.int16)
#     print(int(np.average(np.abs(data))))

# stream.stop_stream()
# stream.close()
# p.terminate()