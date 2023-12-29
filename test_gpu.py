import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Only allocate GPU memory as needed
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Available GPU: {gpu}")
else:
  print("No available GPUs")
