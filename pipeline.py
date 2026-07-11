import os
import argparse
import string
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
import csv

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

    @staticmethod
    def draw_predictions(img, output):
        """
        Draw bounding boxes and predicted text on the image with color coding.
        
        - 'meter' boxes are drawn in GREEN
        - 'nomor_meter' boxes are drawn in ORANGE (BGR)
        
        Returns the annotated image as a numpy array (BGR).
        """
        img_copy = img.copy()
        colors = {
            'meter': (0, 200, 0),
            'nomor_meter': (0, 165, 255)
        }

        for category in ['meter', 'nomor_meter']:
            if output[category]['box'] is None:
                continue
            x1, y1, x2, y2 = output[category]['box']
            text = output[category]['text'] or 'N/A'
            conf = output[category]['conf']
            color = colors[category]

            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            label = f"{category}: {text} ({conf:.2f})"
            font_scale = 0.5
            thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            text_x = x1
            if y1 - label_h - 6 > 0:
                text_y = y1 - 4
                cv2.rectangle(img_copy,
                              (text_x, text_y - label_h - 2),
                              (text_x + label_w, text_y + 2),
                              color, -1)
            else:
                text_y = y2 + label_h + 6
                cv2.rectangle(img_copy,
                              (text_x, text_y - label_h - 2),
                              (text_x + label_w, text_y + 2),
                              color, -1)

            cv2.putText(img_copy, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), thickness)

        return img_copy

    def process(self, image_path, padding=8, conf_threshold=0.25, save_crops_dir=None, output_dir=None):
        """
        Processes a single kilometer photo.

        Args:
            image_path: Path to the input image.
            padding: Padding in pixels around detected bounding boxes before OCR.
            conf_threshold: YOLO confidence threshold.
            save_crops_dir: Optional directory to save cropped text regions.
            output_dir: Optional directory to save the annotated image
                        (with bounding boxes and predicted text drawn on it).

        Returns:
            dict with keys 'meter', 'nomor_meter' (each containing 'text', 'conf', 'box'),
            and 'detected_boxes_count'.
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
            
        # 5. Save annotated image (with bounding boxes and predicted text) if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            annotated = self.draw_predictions(img, output)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
            cv2.imwrite(out_path, annotated)
            print(f"[*] Saved annotated image to {out_path}")
            
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Full Pipeline Kilometer Detector (YOLO + OCR)")
    # Paths
    parser.add_argument('--image_path', required=True,
                        help='Path to a kilometer photo or a directory containing multiple photos')
    parser.add_argument('--yolo_model', default='best.pt', help='Path to the YOLO model (.pt) weight')
    parser.add_argument('--saved_model', required=True, help="Path to the trained OCR model (.pth) weight")
    parser.add_argument('--save_crops_dir', default=None, help="Directory to save cropped boxes (optional)")
    parser.add_argument('--output_dir', default=None,
                        help='Directory to save annotated images (with bboxes and text drawn). '
                             'For single image input, defaults to the same directory as the input image. '
                             'For directory input, defaults to <dirname>_output/')
    
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
    parser.add_argument('--gt_csv', default=None,
                        help='Path to ground truth CSV with columns: Nama file, Stand Meter, Nomor Meter')

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

    # Determine if input is a directory or single file
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    def load_ground_truth(csv_path):
        """Load ground truth from CSV file.

        Expected CSV columns: Nama file, Stand Meter, Nomor Meter
        If a filename has no extension, .jpg is assumed.
        Returns: dict mapping filename -> {'stan_meter': str, 'nomor_meter': str}
        """
        gt_dict = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, 2):  # start from 2 for header row
                filename = os.path.basename(row.get('Nama file', '').strip())
                stan = row.get('Stand Meter', '').strip()
                nomor = row.get('Nomor Meter', '').strip()
                if not filename:
                    print(f"[Warning] CSV row {row_num}: empty filename, skipping")
                    continue
                # If CSV filename has no extension, assume .jpg
                if '.' not in filename:
                    filename = filename + '.jpg'
                gt_dict[filename] = {'stan_meter': stan, 'nomor_meter': nomor}
        return gt_dict

    def _lookup_gt(gt_dict, base_name):
        """Look up filename in gt_dict with flexible extension matching.
        
        Tries exact match first, then falls back to matching without extension
        (handles cases where CSV has .jpg but file is .png, or vice versa).
        """
        if not gt_dict:
            return None
        if base_name in gt_dict:
            return gt_dict[base_name]
        # Fallback: try stripping extension
        name_no_ext = os.path.splitext(base_name)[0]
        return gt_dict.get(name_no_ext)

    # Load ground truth if provided
    gt_dict = {}
    if opt.gt_csv:
        if not os.path.exists(opt.gt_csv):
            print(f"[Error] Ground truth CSV '{opt.gt_csv}' does not exist.")
            exit(1)
        gt_dict = load_ground_truth(opt.gt_csv)
        print(f"[*] Loaded ground truth for {len(gt_dict)} image(s) from '{opt.gt_csv}'")

    def process_single_image(image_path, output_dir, gt_dict=None):
        """Helper to process one image, print results, and optionally compare with ground truth.

        Returns:
            tuple: (result_dict, metrics_dict or None)
        """
        res = pipeline.process(
            image_path=image_path,
            padding=opt.padding,
            conf_threshold=opt.yolo_conf,
            save_crops_dir=opt.save_crops_dir,
            output_dir=output_dir
        )

        base_name = os.path.basename(image_path)

        print("\n" + "=" * 50)
        print(f"Image Path     : {image_path}")
        print(f"Boxes Detected : {res['detected_boxes_count']}")
        print("-" * 50)

        meter_val = res['meter']['text']
        meter_conf = res['meter']['conf']
        print(f"Stand Meter    : {meter_val if meter_val else 'N/A'} (Conf: {meter_conf:.4f})")

        no_meter_val = res['nomor_meter']['text']
        no_meter_conf = res['nomor_meter']['conf']
        print(f"Nomor Meter    : {no_meter_val if no_meter_val else 'N/A'} (Conf: {no_meter_conf:.4f})")

        # Ground truth comparison
        result_metrics = None
        gt_entry = _lookup_gt(gt_dict, base_name)
        if gt_entry is not None:
            from nltk.metrics.distance import edit_distance

            gt_meter = gt_entry['stan_meter']
            gt_nomor = gt_entry['nomor_meter']

            pred_meter = res['meter']['text'] or ''
            pred_nomor = res['nomor_meter']['text'] or ''

            meter_match = pred_meter == gt_meter
            nomor_match = pred_nomor == gt_nomor
            meter_ed = edit_distance(pred_meter, gt_meter)
            nomor_ed = edit_distance(pred_nomor, gt_nomor)

            print("-" * 50)
            print(f"GT Stand Meter : {gt_meter}")
            print(f"GT Nomor Meter : {gt_nomor}")
            print(f"Meter Match    : {'✓' if meter_match else '✗'} (Edit Dist: {meter_ed})")
            print(f"Nomor Match    : {'✓' if nomor_match else '✗'} (Edit Dist: {nomor_ed})")

            result_metrics = {
                'file': base_name,
                'stan_meter_gt': gt_meter,
                'stan_meter_pred': pred_meter,
                'nomor_meter_gt': gt_nomor,
                'nomor_meter_pred': pred_nomor,
                'meter_match': meter_match,
                'nomor_match': nomor_match,
                'meter_ed': meter_ed,
                'nomor_ed': nomor_ed,
                'both_correct': meter_match and nomor_match,
            }
        elif gt_dict is not None:
            print(f"[Warning] '{base_name}' not found in ground truth CSV")

        print("=" * 50 + "\n")
        return res, result_metrics

    try:
        if os.path.isdir(opt.image_path):
            # --- Batch processing: directory input ---
            input_dir = opt.image_path.rstrip('/')
            output_dir = input_dir + '_output'

            image_files = sorted([
                f for f in os.listdir(input_dir)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ])

            if not image_files:
                print(f"[Warning] No image files found in directory: {input_dir}")
                exit(1)

            print(f"[*] Found {len(image_files)} image(s) in '{input_dir}'")
            print(f"[*] Annotated images will be saved to '{output_dir}'\n")
            os.makedirs(output_dir, exist_ok=True)

            summary = []
            all_metrics = []
            for img_file in image_files:
                img_path = os.path.join(input_dir, img_file)
                print(f"[Processing] {img_file} ...")
                try:
                    res, metrics = process_single_image(img_path, output_dir, gt_dict)
                    summary.append({
                        'file': img_file,
                        'meter': res['meter']['text'] or 'N/A',
                        'meter_conf': res['meter']['conf'],
                        'nomor_meter': res['nomor_meter']['text'] or 'N/A',
                        'nomor_meter_conf': res['nomor_meter']['conf'],
                    })
                    if metrics:
                        all_metrics.append(metrics)
                except Exception as e:
                    print(f"[Error] Failed to process {img_file}: {e}")

            # Print summary table
            print("=" * 70)
            print(f"{'File':30s} {'Stand Meter':20s} {'Nomor Meter':20s}")
            print("=" * 70)
            for s in summary:
                m = f"{s['meter']} ({s['meter_conf']:.2f})"
                nm = f"{s['nomor_meter']} ({s['nomor_meter_conf']:.2f})"
                print(f"{s['file']:30s} {m:20s} {nm:20s}")
            print("=" * 70)
            print(f"Processed {len(summary)} image(s). Annotated images saved to: {output_dir}")

            # Print metrics summary if ground truth was provided
            if all_metrics:
                n = len(all_metrics)
                meter_correct = sum(1 for m in all_metrics if m['meter_match'])
                nomor_correct = sum(1 for m in all_metrics if m['nomor_match'])
                both_correct = sum(1 for m in all_metrics if m['both_correct'])
                avg_meter_ed = sum(m['meter_ed'] for m in all_metrics) / n
                avg_nomor_ed = sum(m['nomor_ed'] for m in all_metrics) / n

                print("\n" + "=" * 60)
                print("           METRICS SUMMARY")
                print("=" * 60)
                print(f"Total Images with GT    : {n}")
                print(f"Stand Meter Accuracy    : {meter_correct/n*100:.1f}% ({meter_correct}/{n})")
                print(f"Nomor Meter Accuracy    : {nomor_correct/n*100:.1f}% ({nomor_correct}/{n})")
                print(f"Overall Accuracy        : {both_correct/n*100:.1f}% ({both_correct}/{n})")
                print(f"Avg Meter Edit Distance : {avg_meter_ed:.2f}")
                print(f"Avg Nomor Edit Distance : {avg_nomor_ed:.2f}")
                print("=" * 60 + "\n")

        else:
            # --- Single image processing ---
            if opt.output_dir:
                output_dir = opt.output_dir
            else:
                output_dir = os.path.dirname(os.path.abspath(opt.image_path))

            process_single_image(opt.image_path, output_dir, gt_dict)

    except Exception as e:
        print(f"[Error] Failed to process image: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
