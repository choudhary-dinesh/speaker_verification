import glob
import os
import shutil
import random

def prepare_test_train_file_list(voxceleb_path,  train_ratio):
  all_file_paths = []
  for f in glob.glob(voxceleb_path+"/*/*/*") :
    destination_folder = "/".join(f.split("/")[:-2])
    dest_path = os.path.join(destination_folder , "_".join(f.split("/")[-2:]))
    shutil.move(f,dest_path)
    all_file_paths.append(dest_path)

  test_ratio = 1 - train_ratio  # Remaining for testing

  num_train_samples = int(train_ratio * len(all_file_paths))
  num_test_samples = len(all_file_paths) - num_train_samples

  all_file_paths_train = sorted(random.sample(all_file_paths, num_train_samples))
  all_file_paths_test = sorted([path for path in all_file_paths if path not in all_file_paths_train])

  print("Train data size:", len(all_file_paths_train))
  print("Test data size:", len(all_file_paths_test))
  
  return all_file_paths_train, all_file_paths_test
