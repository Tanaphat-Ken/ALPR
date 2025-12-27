import io
import torch.nn as nn
import torch.nn.init as init
from PIL import Image

def init_weights(modules):
  for m in modules:
    if isinstance(m, nn.Conv2d):
      init.xavier_uniform_(m.weight.data)
      if m.bias is not None:
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
      m.weight.data.fill_(1)
      m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
      m.weight.data.normal_(0, 0.01)
      m.bias.data.zero_()

def find_largest_bbox(detections):
  largest_box = None
  max_area = 0

  for detection in detections:
    x, y, w, h = detection['bbox'][0]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    area = (x2 - x1) * (y2 - y1)

    if area > max_area:
      max_area = area
      largest_box = [x1, y1, x2, y2]

  return largest_box

def crop_4point_image(image, points):
  x_values = [p[0] for p in points]
  y_values = [p[1] for p in points]

  left = min(x_values)
  top = min(y_values)
  right = max(x_values)
  bottom = max(y_values)

  cropped_image = image.crop((left, top, right, bottom))
  return cropped_image

def image_open_and_verify(file_content):
  image = Image.open(io.BytesIO(file_content)).convert("RGB")
  image.verify()
  return image

def convert_2_to_4_point(bbox):
  if bbox is None: return None

  xmin, ymin, xmax, ymax = bbox

  return [
    (xmin, ymin),
    (xmax, ymin),
    (xmax, ymax),
    (xmin, ymax)
  ]