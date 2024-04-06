from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix
import numpy as np
def calculate_metrics(true_labels, predicted_labels):
  accuracy = accuracy_score(true_labels, predicted_labels)
  precision = precision_score(true_labels, predicted_labels,average='macro')
  recall = recall_score(true_labels, predicted_labels,average='macro')
  cm = confusion_matrix(true_labels, predicted_labels)
  return accuracy, precision, recall,cm

def calculate_far_frr_multiclass(confusion_matrix, class_index):
  tn = np.delete(np.delete(confusion_matrix, class_index, axis=0), class_index, axis=1).sum()
  fp = np.delete(confusion_matrix[class_index, :], class_index).sum()
  fn = np.delete(confusion_matrix[:, class_index], class_index).sum()
  tp = confusion_matrix[class_index, class_index]
  far = fp / (fp + tn)
  frr = fn / (fn + tp)
  return round(far*100,2), round(frr*100,2)
