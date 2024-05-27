import torch
from torchvision.transforms import functional as F
from PIL import Image
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
import math

def load_model(model_path, device):
    import torchvision
    weights = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 51  # 50 ship classes + background
    in_features = weights.roi_heads.box_predictor.cls_score.in_features
    weights.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model = weights.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def predict_image(model, image_path, device, confidence_threshold=0.5):
    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        predictions = model(img_tensor)

    # Process the predictions
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()

    # Filter out low-confidence detections
    high_conf_indices = pred_scores > confidence_threshold
    pred_boxes = pred_boxes[high_conf_indices]
    pred_labels = pred_labels[high_conf_indices]
    pred_scores = pred_scores[high_conf_indices]

    return pred_boxes, pred_labels, pred_scores

def display_predictions(image_path, boxes, labels, scores, class_names):
    img = Image.open(image_path).convert("RGB")
    fig = go.Figure()
    fig.add_trace(go.Image(z=np.array(img)))

    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        fig.add_shape(type='rect',
                      x0=xmin, y0=ymin, x1=xmax, y1=ymax,
                      line=dict(color='red', width=2))
        fig.add_annotation(x=xmin, y=ymin,
                           text=f"{class_names[label-1]}: {score:.2f}",
                           showarrow=False,
                           bgcolor='yellow', opacity=0.5)

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig

def parse_filename(filename):
    # Extract coordinates and scale from filename using regex
    match = re.match(r"(\d+\.\d+)_(\d+\.\d+)_1.(\d+)", filename)
    if match:
        lat, lon, scale = match.groups()
        return float(lat), float(lon), int(scale)
    return None, None, None

def calculate_geo_coordinates(box, img_size, center_lat, center_lon, scale, direction):
    # Вычислить координаты центра обьекта
    xmin, ymin, xmax, ymax = box
    box_center_x = (xmin + xmax) / 2
    box_center_y = (ymin + ymax) / 2

    # Разрешения изображения
    img_width, img_height = img_size

    # Вычисление географических координат обьекта
    # Смещение относительно центра
    y_offset = (box_center_y - img_height / 2)
    x_offset = (box_center_x - img_width / 2)

    # Реальное смещение относительно центра
    lat_real_offset = y_offset * scale
    lon_real_offset = x_offset * scale
    
    # Вычисление смещения учитывая направление изображения
    if direction == 'north':
        lat_real_offset = -lat_real_offset
    elif direction == 'south':
        lat_real_offset = lat_real_offset
    elif direction == 'west':
        lon_real_offset = -lon_real_offset
    elif direction == 'east':
        lon_real_offset = lon_real_offset
    else:
        raise ValueError("Invalid direction")
    
    # Преобразование в градусы
    lat_deg = lat_real_offset / (111000 *math.cos(center_lat))
    lon_deg = lon_real_offset / (111000 *math.cos(center_lon))

    # Итоговые координаты
    lat = center_lat + lat_deg
    lon = center_lon + lon_deg

    return lat, lon

def create_xml_annotation(xml_path, image_path, boxes, labels, scores, class_names, lat, lon, scale, direction):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = image_path
    ET.SubElement(root, "path").text = image_path

    size = ET.SubElement(root, "size")
    img = Image.open(image_path)
    width, height = img.size
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    coordinates = ET.SubElement(root, "coordinates")
    ET.SubElement(coordinates, "latitude").text = str(lat)
    ET.SubElement(coordinates, "longitude").text = str(lon)
    ET.SubElement(coordinates, "scale").text = str(scale)

    for box, label, score in zip(boxes, labels, scores):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = class_names[label - 1]
        ET.SubElement(obj, "score").text = str(score)
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(box[0]))
        ET.SubElement(bndbox, "ymin").text = str(int(box[1]))
        ET.SubElement(bndbox, "xmax").text = str(int(box[2]))
        ET.SubElement(bndbox, "ymax").text = str(int(box[3]))
        
        # Calculate and add geographical coordinates of the bounding box center
        geo_lat, geo_lon = calculate_geo_coordinates(box, (width, height), lat, lon, scale, direction)
        geo = ET.SubElement(obj, "geo")
        ET.SubElement(geo, "latitude").text = str(geo_lat)
        ET.SubElement(geo, "longitude").text = str(geo_lon)

    # Pretty-print the XML
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml_as_string = reparsed.toprettyxml(indent="  ")

    with open(xml_path, "w") as f:
        f.write(pretty_xml_as_string)
