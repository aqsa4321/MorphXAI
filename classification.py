import json

def calculate_iou(box1, box2):
    """Both boxes in [x, y, w, h] format"""
    x1_1, y1_1, x2_1, y2_1 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    x1_2, y1_2, x2_2, y2_2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    
    x1, y1 = max(x1_1, x1_2), max(y1_1, y1_2)
    x2, y2 = min(x2_1, x2_2), min(y2_1, y2_2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1, area2 = box1[2] * box1[3], box2[2] * box2[3]
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

# Load data
with open('/home/aqsa/Final_Dataset/annotations_shape/val_combined_shape.json', 'r') as f:
    gt_data = json.load(f)
with open('/home/aqsa/shape_new/RT-DETRv3-shape0.5/bbox.json', 'r') as f:
    pred_data = json.load(f)

# Get predictions with morphological data
morph_traits = ['shape_type', 'curvature', 'dot_count', 'flagellum_present', 'location']
preds_with_morph = [p for p in pred_data if all(f'{trait}_class' in p for trait in morph_traits)]
print(f"Total predictions with morphology: {len(preds_with_morph)}")

# Filter by confidence
confidence_threshold = 0.5
filtered_preds = [p for p in preds_with_morph if p['score'] >= confidence_threshold]
print(f"High-confidence predictions with morphology: {len(filtered_preds)}")

# Initialize stats
stats = {trait: {'correct': 0, 'total': 0} for trait in morph_traits}
stats['complete_match'] = {'correct': 0, 'total': 0}

# Match predictions to ground truth
iou_threshold = 0.5
matched_pairs = 0

for pred in filtered_preds:
    pred_bbox = pred['bbox']
    pred_image_id = pred['image_id']
    
    # Find GT annotations for this image
    gt_annotations = [ann for ann in gt_data['annotations'] if ann['image_id'] == pred_image_id]
    
    # Find best matching GT annotation
    best_iou = 0
    best_gt = None
    
    for gt_ann in gt_annotations:
        gt_bbox = gt_ann['bbox']
        iou = calculate_iou(pred_bbox, gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best_gt = gt_ann
    
    # If IoU is above threshold, evaluate morphology
    if best_iou >= iou_threshold and best_gt is not None:
        matched_pairs += 1
        
        # Compare morphological traits
        complete_match = True
        for trait in morph_traits:
            pred_value = pred[f'{trait}_class']
            gt_value = best_gt['morphology'][trait]
            
            stats[trait]['total'] += 1
            if pred_value == gt_value:
                stats[trait]['correct'] += 1
            else:
                complete_match = False
        
        # Track complete morphology matches
        stats['complete_match']['total'] += 1
        if complete_match:
            stats['complete_match']['correct'] += 1

# Print results
print(f"\nMatched pairs: {matched_pairs} (IoU >= {iou_threshold})")
print("\n=== MORPHOLOGICAL CLASSIFICATION RESULTS ===")
for trait in morph_traits + ['complete_match']:
    if stats[trait]['total'] > 0:
        accuracy = stats[trait]['correct'] / stats[trait]['total']
        print(f"{trait}: {stats[trait]['correct']}/{stats[trait]['total']} = {accuracy:.3f}")
    else:
        print(f"{trait}: No matches found")