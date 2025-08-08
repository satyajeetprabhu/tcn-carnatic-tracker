import numpy as np
import matplotlib.pyplot as plt
import librosa
import IPython.display as ipd

def plot_audio(audio, annotations, detections, sr):
  hop_length = 512

  # we will plot only 20 seconds otherwise it is too hard to visualize
  audio = audio[:sr*20]
  annotations = annotations[annotations <= 20]
  detections = detections[detections <= 20]

  spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=hop_length)), ref=np.max)
  librosa.display.specshow(spec, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time')
  plt.title(f"Log-frequency power spectrogram of track")
  plt.colorbar(format="%+2.f dB")
  # plot annotations in the upper part
  plt.vlines(annotations, hop_length * 2, sr / 2, linestyles='dotted', color='w')
  plt.text(7, hop_length * 1.65, 'Annotations (above)', color='w', fontsize=12)
  # plot detections in the lower part
  plt.vlines(detections, 0, hop_length, linestyles='dotted', color='w')
  plt.text(7, hop_length * 1.1, 'Detections (below)', color='w', fontsize=12)
  plt.show()

 
def plot_spec(audio, beat_ann, db_ann, beat_det, db_det, sr):
  hop_length = 512

  # we will plot only 20 seconds otherwise it is too hard to visualize
  audio = audio[:sr*20]
  beat_ann = beat_ann[beat_ann <= 20]
  db_ann = db_ann[db_ann <= 20]
  beat_det = beat_det[beat_det <= 20]
  db_det = db_det[db_det <= 20]

  spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=hop_length)), ref=np.max)
  librosa.display.specshow(spec, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time')
  plt.title(f"Log-frequency power spectrogram of track")
  plt.colorbar(format="%+2.f dB")
  # plot annotations in the upper part
  plt.vlines(beat_ann, hop_length * 2, sr / 2, linestyles='dotted', color='w')
  plt.vlines(db_ann, hop_length * 2, sr / 2, color='w')
  plt.text(7, hop_length * 1.65, 'Annotations (above)', color='w', fontsize=12)
  # plot detections in the lower part
  plt.vlines(beat_det, 0, hop_length, linestyles='dotted', color='w')
  plt.vlines(db_det, 0, hop_length, color='w')
  plt.text(7, hop_length * 1.1, 'Detections (below)', color='w', fontsize=12)
  plt.show()
    
'''
song = test.ids[0]
det = detections[song]
track = tracks[song]
audio, sr = track.audio
hop_length = 512

spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=hop_length)), ref=np.max)
librosa.display.specshow(spec, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time')
plt.title(f'Log-frequency power spectrogram of track: "{song}"')
plt.colorbar(format="%+2.f dB")
# plot annotations in the upper part
plt.vlines(track.beats.times, hop_length * 2, sr / 2, linestyles='dotted', color='w')
plt.vlines(track.beats.times[track.beats.positions == 1], hop_length * 2, sr / 2, color='w')
plt.text(7, hop_length * 1.65, 'Annotations (above)', color='w', fontsize=12)
# plot detections in the lower part
plt.vlines(det['downbeats'][:, 0], 0, hop_length, linestyles='dotted', color='w')
plt.vlines(det['downbeats'][det['downbeats'][:, 1] == 1][:, 0], 0, hop_length, color='w')
plt.text(7, hop_length * 1.1, 'Detections (below)', color='w', fontsize=12)
plt.show()
'''

def plot_activations(act, title='Activations'):

    plt.figure(figsize=(10, 3))
    plt.plot(act, color='blue')
    plt.title(title)
    plt.xlabel('Time (frames)')
    plt.ylabel('Activation')
    plt.grid()

    plt.tight_layout()
    plt.show()