import os
import argparse
import string
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import AlignCollate
from model import Model

class KilometerPipeline:
    """
    End-to-End Pipeline for Kilometer Images.
    1. Detects text bounding boxes using YOLO.
    2. Crops the bounding boxes with padding.
    3. Recognizes text using deep-text-recognition-benchmark (CRNN) model.
    4. Separates findings into 'meter' (top box) and 'nomor_meter' (bottom box).
    """
    def __init__(self, yolo_model_path, ocr_model_path, ocr_opt):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[*] Running pipeline on device: {self.device}")
        
        # 1. Load YOLO Model
        print(f"[+] Loading YOLO model from {yolo_model_path}...")
        self.yolo_model = YOLO(yolo_model_path)
        
        # 2. Configure OCR Model Options
        self.ocr_opt = ocr_opt
        if 'CTC' in ocr_opt.Prediction:
            self.converter = CTCLabelConverter(ocr_opt.character)
        else:
            self.converter = AttnLabelConverter(ocr_opt.character)
        ocr_opt.num_class = len(self.converter.character)

        if ocr_opt.rgb:
            ocr_opt.input_channel = 3
        else:
            ocr_opt.input_channel = 1
            
        # 3. Load OCR Model
        print(f"[+] Loading OCR model from {ocr_model_path}...")
        self.ocr_model = Model(ocr_opt)
        self.ocr_model = torch.nn.DataParallel(self.ocr_model).to(self.device)
        self.ocr_model.load_state_dict(torch.load(ocr_model_path, map_location=self.device))
        self.ocr_model.eval()
        
        # 4. Prepare OCR Collator
        self.align_collate = AlignCollate(imgH=ocr_opt.imgH, imgW=ocr_opt.imgW, keep_ratio_with_pad=ocr_opt.PAD)

    def predict_crop(self, pil_image):
        """
        Run OCR model prediction on a single cropped PIL image.
        """
        image_tensors, _ = self.align_collate([(pil_image, '')])
        batch_size = image_tensors.size(0)
        image = image_tensors.to(self.device)
        
        # For max length prediction
        length_for_pred = torch.IntTensor([self.ocr_opt.batch_max_length] * batch_size).to(self.device)
        text_for_pred = torch.LongTensor(batch_size, self.ocr_opt.batch_max_length + 1).fill_(0).to(self.device)

        with torch.no_grad():
            if 'CTC' in self.ocr_opt.Prediction:
                preds = self.ocr_model(image, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, preds_size)
            else:
                preds = self.ocr_model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            
            pred = preds_str[0]
            pred_max_prob = preds_max_prob[0]
            
            if 'Attn' in self.ocr_opt.Prediction:
                pred_EOS = pred.find('[s]')
                if pred_EOS != -1:
                    pred = pred[:pred_EOS]
                    pred_max_prob = pred_max_prob[:pred_EOS]

            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
            except Exception:
                confidence_score = 0.0
                
        return pred, confidence_score

    def process(self, image_path, padding=8, conf_threshold=0.25, save_crops_dir=None):
        """
        Processes a single kilometer photo.
        Returns a dictionary containing 'meter', 'nomor_meter', and their confidence scores.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")
            
        # 1. Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image at: {image_path}")
        h_img, w_img, _ = img.shape
        
        # 2. Detect with YOLO
        results = self.yolo_model.predict(image_path, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
        
        output = {
            'meter': {'text': None, 'conf': 0.0, 'box': None},
            'nomor_meter': {'text': None, 'conf': 0.0, 'box': None},
            'detected_boxes_count': len(boxes)
        }
        
        if len(boxes) == 0:
            print("[Warning] No bounding boxes detected by YOLO.")
            return output
            
        # 3. Sort boxes by Y-coordinate (top to bottom)
        # b[1] is y1 coordinate
        sorted_boxes = sorted(boxes, key=lambda b: b[1])
        
        # Classify based on position
        if len(sorted_boxes) >= 2:
            # Top box is stand meter, bottom box is nomor meter
            pairs = [
                (sorted_boxes[0], 'meter'),
                (sorted_boxes[-1], 'nomor_meter')
            ]
            if len(sorted_boxes) > 2:
                print(f"[Info] Detected {len(sorted_boxes)} boxes. Using top-most for 'meter' and bottom-most for 'nomor_meter'.")
        else:
            # Only 1 box detected
            # Decide based on vertical center of the image
            box = sorted_boxes[0]
            y_center = (box[1] + box[3]) / 2.0
            category = 'meter' if y_center < (h_img / 2.0) else 'nomor_meter'
            print(f"[Warning] Only 1 box detected by YOLO. Placed in '{category}' based on vertical position.")
            pairs = [(box, category)]
            
        # 4. Crop, Predict OCR
        if save_crops_dir:
            os.makedirs(save_crops_dir, exist_ok=True)
            
        for box, category in pairs:
            x1, y1, x2, y2 = map(int, box)
            
            # Apply padding
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w_img, x2 + padding)
            y2_pad = min(h_img, y2 + padding)
            
            # OpenCV crop
            crop_img = img[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Save crops for visual verification if requested
            if save_crops_dir:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = os.path.join(save_crops_dir, f"{base_name}_{category}.jpg")
                cv2.imwrite(save_path, crop_img)
                print(f"[*] Saved cropped image to {save_path}")
            
            # Convert to PIL for OCR model
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)
            
            if not self.ocr_opt.rgb:
                pil_crop = pil_crop.convert('L')
                
            # Run OCR prediction
            pred_text, conf_score = self.predict_crop(pil_crop)
            
            output[category] = {
                'text': pred_text,
                'conf': conf_score,
                'box': [x1, y1, x2, y2]
            }
            
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Full Pipeline Kilometer Detector (YOLO + OCR)")
    # Paths
    parser.add_argument('--image_path', required=True, help='Path to the original kilometer photo')
    parser.add_argument('--yolo_model', default='best.pt', help='Path to the YOLO model (.pt) weight')
    parser.add_argument('--saved_model', required=True, help="Path to the trained OCR model (.pth) weight")
    parser.add_argument('--save_crops_dir', default=None, help="Directory to save cropped boxes (optional)")
    
    # OCR Options (configured to match the model architecture)
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=64, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=200, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    
    # OCR Model Architecture Settings (Defaults mapped to training config)
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='CustomAttentionCNN', 
                        help='FeatureExtraction stage. VGG|RCNN|ResNet|CustomAttentionCNN|CustomCBAMCNN|ResNet_CBAM')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
    
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    
    # YOLO Options
    parser.add_argument('--yolo_conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--padding', type=int, default=8, help='Padding for cropping text boxes')

    opt = parser.parse_args()

    # Check model paths
    if not os.path.exists(opt.yolo_model):
        print(f"[Error] YOLO model path '{opt.yolo_model}' does not exist.")
        exit(1)
    if not os.path.exists(opt.saved_model):
        print(f"[Error] OCR model path '{opt.saved_model}' does not exist.")
        exit(1)

    # Initialize Pipeline
    pipeline = KilometerPipeline(
        yolo_model_path=opt.yolo_model,
        ocr_model_path=opt.saved_model,
        ocr_opt=opt
    )
    
    # Process Image
    try:
        res = pipeline.process(
            image_path=opt.image_path,
            padding=opt.padding,
            conf_threshold=opt.yolo_conf,
            save_crops_dir=opt.save_crops_dir
        )
        
        # Print Results
        print("\n" + "="*50)
        print(f"Image Path     : {opt.image_path}")
        print(f"Boxes Detected : {res['detected_boxes_count']}")
        print("-"*50)
        
        meter_val = res['meter']['text']
        meter_conf = res['meter']['conf']
        print(f"Stand Meter    : {meter_val if meter_val else 'N/A'} (Conf: {meter_conf:.4f})")
        
        no_meter_val = res['nomor_meter']['text']
        no_meter_conf = res['nomor_meter']['conf']
        print(f"Nomor Meter    : {no_meter_val if no_meter_val else 'N/A'} (Conf: {no_meter_conf:.4f})")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"[Error] Failed to process image: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
