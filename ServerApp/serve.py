import base64
from pathlib import Path
import os
import argparse
import cv2
import numpy as np

import torch
# import torch.backends.cudnn as cudnn
from models.common import DetectPTBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors, save_one_box

# Flask utils
from flask import Flask, request, jsonify

IMG_SIZE = (640, 640)
STRIDE = 32
device = None

# Define a flask app
app = Flask(__name__)


@app.route('/predict', methods=["POST", "GET"])
def upload():

        # =========== Load Image ===========
        decoded_image = base64.b64decode(request.json['image'])
        im0 = cv2.imdecode(np.frombuffer(decoded_image, np.uint8), cv2.IMREAD_COLOR)
        
        # =========== Preprocess Image ===========
        im = letterbox(im0, IMG_SIZE[0], stride=32, auto=True)[0] # Resize and padding
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im) # (3, 640, 640)
        im = torch.from_numpy(im).to(device).float().unsqueeze(0) # Convert image to tensor
        im /= 255  # 0 - 255 to 0.0 - 1.0
        
        ### =========== Inference ===========
        pred = model(im) # (1, 25200, 85) (center x, center y, width, height, conf, 80 class prob)
        # print(pred[0].numpy().flatten()[:10])
        
        ### =========== Post-processing ===========
        det = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)[0]  # (N, 6)  (x1, y1, x2, y2, conf, cls)
        
        results_list = []
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det:
                result_dict = {
                    "box": [float(p) for p in xyxy],
                    "score": float(conf),
                    "cls": int(cls),
            }
                results_list.append(result_dict)
        return jsonify(results_list)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model path')
    opt = parser.parse_args()

    # 5. Load the model once at startup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DetectPTBackend(opt.weights, device=device)

    port = os.environ.get('PORT', 8008)
    app.run(debug=False, host='0.0.0.0', port=port)