import numpy as np
from scipy.io.wavfile import read
from feature import extract_features
from sklearn.mixture import GaussianMixture as GMM
import _pickle as cPickle

def train_gmm(file_paths, dest):
  speaker_id_prev = None
  embedding = np.asarray(())
  all_train_dict = {item.split("/")[-2] : list(filter(lambda x: item.split("/")[-2] in x, file_paths))  for item in file_paths}
  for speaker,path_list in all_train_dict.items():
    for path in path_list :
      path = path.strip()
      # read the audio
      sample_rate,audio = read(path)
      # Extract Features(MFCC & delta MFCC)
      feature_vector =  extract_features(audio,sample_rate)
      if embedding.size == 0:
          embedding = feature_vector
      else:
          embedding = np.vstack((embedding, feature_vector))
    print(f"Training model for {speaker}")
    gmm = GMM(n_components = 5, covariance_type='diag',n_init = 3)
    gmm.fit(embedding)
    picklefile = speaker +".gmm"
    cPickle.dump(gmm,open(dest + picklefile,'wb'))
    print ('GMM trainned for speaker:',picklefile)
    embedding = np.asarray(())
