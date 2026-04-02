import numpy as np
# RANDOM_SEED
RANDOM = np.random.RandomState(42)

def openAudioFile(path, sample_rate=48000, offset=0.0, duration=None):    
    
    # Open file with librosa (uses ffmpeg or libav)
    import librosa

    sig, rate = librosa.load(path, sr=sample_rate, offset=offset, duration=duration, mono=True, res_type='kaiser_fast')

    return sig, rate

def saveSignal(sig, fname):

    import soundfile as sf
    sf.write(fname, sig, 48000, 'PCM_16')

def noise(sig, shape, amount=None):

    # Random noise intensity
    if amount == None:
        amount = RANDOM.uniform(0.1, 0.5)

    # Create Gaussian noise
    try:
        noise = RANDOM.normal(min(sig) * amount, max(sig) * amount, shape)
    except:
        noise = np.zeros(shape)

    return noise.astype('float32')

def cropCenter(sig, rate, seconds):

    # Crop signal to center
    if len(sig) > int(seconds * rate):
        start = int((len(sig) - int(seconds * rate)) / 2)
        end = start + int(seconds * rate)
        sig = sig[start:end]
    
    # Pad with noise
    elif len(sig) < int(seconds * rate):
        sig = np.hstack((sig, noise(sig, (int(seconds * rate) - len(sig)), 0.5)))

    return sig