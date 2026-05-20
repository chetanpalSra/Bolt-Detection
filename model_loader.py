import torch
from ultralytics import YOLO
import os

def load_model(model_path):
  """Loads the YOLO model and determines the best hardware accelerator."""
  if not os.path.exists(model_path):
    return None

  #Load model -->
  model = YOLO(model_path)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  return model,device

def run_inference(model,img_array,conf,iou,device):
  """Executes prediction using the chosen thresholds."""
  results = model.predict(
      source = img_array,
      conf = conf,
      iou = iou,
      device = device
  )
  return results[0]
