from sklearn import preprocessing
import python_speech_features as mfcc
import numpy as np
def calculate_delta_mfcc(mfcc_feature):
  rows,cols = mfcc_feature.shape
  deltas = np.zeros((rows,20))
  N = 2
  for i in range(rows):
    index = []
    j = 1
    while j <= N:
      if i-j < 0:
        first =0
      else:
        first = i-j
      if i+j > rows-1:
        second = rows-1
      else:
        second = i+j
      index.append((second,first))
      j+=1
    deltas[i] = (mfcc_feature[index[0][0]]-mfcc_feature[index[0][1]]
                 + (2 * (mfcc_feature[index[1][0]]-mfcc_feature[index[1][1]])) ) / 10
  return deltas

def extract_features(audio,sample_rate):
    mfcc_feature = mfcc.mfcc(audio,sample_rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta_mfcc(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta))
    return combined
