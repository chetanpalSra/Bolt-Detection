def analyze_detections(yolo_result,expected_count):
  """Analyzes the YOLO results, calculates counts, anomalies, and confidence metrics."""
  total_detected = len(yolo_result.boxes)
    # Simple logic to find anomalies/missed bolts
    # (Anomalies are often defined as counts outside of an expected range)
    # Here, we count how far off we are from expected.
  anomalies = max(0,expected_count - total_detected)

  if total_detected > 0:
      confs = yolo_result.boxes.conf.cpu().numpy()
      avg_conf = confs.mean()
  else:
      avg_conf = 0.0

  return total_detected,anomalies,avg_conf
