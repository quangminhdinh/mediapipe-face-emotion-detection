from .encode import extract_features

def log_steps(path, i, total, label, log_interval=100):
  if (i % log_interval == 0):
    print(f"{label}: {i}/{total}")
  return extract_features(path)
