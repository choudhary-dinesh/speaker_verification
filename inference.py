from tqdm import tqdm 
import os
import _pickle as cPickle
from scipy.io.wavfile import read
from feature import extract_features
import numpy as np

def predict_speaker(file_paths,dest):
  gmm_files = [os.path.join(dest,fname) for fname in os.listdir(dest) if fname.endswith('.gmm')]
  models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
  speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]
  print("Total Speakers : ", len(speakers))
  print("Total GMM models : ", len(gmm_files))
  results  = []
  for each in tqdm(file_paths):
    path=each.strip()
    sr,audio = read(path)
    feature_vector   = extract_features(audio,sr)
    log_likelihood = np.zeros(len(models))
    for i in range(len(models)):
      gmm = models[i]  #checking with each model one by one
      pred_scores = np.array(gmm.score(feature_vector))
      log_likelihood[i] = pred_scores.sum()
    pred =np.argmax(log_likelihood)
    results.append([each, np.array(log_likelihood), pred])
  return results,speakers
