# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import itertools
import paddle.nn.functional as F
import paddle
from ppdet.metrics.json_results import get_det_res, get_det_poly_res, get_seg_res, get_solov2_segm_res, get_keypoint_res, get_pose3d_res
from ppdet.metrics.map_utils import draw_pr_curve

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

def generate_morphology_description(detection_data):
    """
    Generate natural language description from morphological predictions
    """

    print(f"DEBUG Report - Input data: {detection_data}")
    print(f"DEBUG Report - Score: {detection_data.get('score', 'Missing')}")
    print(f"DEBUG Report - BBox: {detection_data.get('bbox', 'Missing')}")
    print(f"DEBUG Report - Shape: {detection_data.get('shape_type_class', 'Missing')}")
    print("-" * 30)


    # Map category IDs to species names
    species_mappings = { 
        1: 'Leishmania spp.',
        2: 'Trypanosoma brucei',
        3: 'Trypanosoma cruzi',
        # Add other category IDs as needed
    }
    
    # Get species name from category_id
    category_id = detection_data.get('category_id', 0)
    species_name = species_mappings.get(category_id, 'Unknown parasite')


    # Define mappings from class indices to text
    morphology_mappings = {
        'shape_type_class': {
            0: 'oval-shaped',           # ring (Babesia)
            1: 'elongated',             # elongated (T. brucei, T. cruzi, Leishmania)
            2: 'amoeboid',              # amoeboid 
            3: 'fusiform',              # pear-shaped (pyriform)
            4: 'crescent-shaped',       # crescent (Toxoplasma)
            5: 'undefined shape'        # none
        },
        'curvature_class': {            # ← NEW: Replace size_range_class
            0: 'straight',              # straight (Leishmania)
            1: 'C-shaped',       # slightly curved (T. brucei)
            2: 'S-shaped',         # C/S-shaped (T. cruzi, Toxoplasma)
            3: 'no curvature'           # none (round like Babesia)
        },
        'dot_count_class': {            # ← NEW: Replace nucleus_count_class
            0: 'no visible dots',       # 0 dots (Babesia, Toxoplasma)
            1: 'one visible dot',       # 1 dot (T. brucei - nucleus)
            2: 'two visible dots',      # 2 dots (T. cruzi, Leishmania - nucleus + kinetoplast)
            3: 'multiple dots'          # 3+ dots
        },
        'flagellum_present_class': {    # ← KEEP (unchanged)
            0: 'without flagella',
            1: 'with flagella'
        },
        'location_class': {             # ← KEEP (unchanged)
            0: 'premature',
            1: 'mature'
        }
    }
    # Extract morphological features with confidence
    description_parts = [f"Detected: {species_name}"]
    confidence_parts = []
    
    # print(detection_data)
    if 'shape_type_class' in detection_data:
        shape_idx = int(detection_data['shape_type_class'])
        shape_desc = morphology_mappings['shape_type_class'].get(shape_idx, 'unknown shape')
        shape_conf = detection_data.get('shape_type_prob', 0.0)
        description_parts.append(f"This parasite is {shape_desc}")
        confidence_parts.append(f"shape: {shape_conf:.2f}")
    
    if 'curvature_class' in detection_data:                    # ← CHANGE from 'size_range_class'
        curvature_idx = int(detection_data['curvature_class'])
        curvature_desc = morphology_mappings['curvature_class'].get(curvature_idx, 'unknown curvature')
        curvature_conf = detection_data.get('curvature_prob', 0.0)
        description_parts.append(f"with {curvature_desc} morphology")
        confidence_parts.append(f"curvature: {curvature_conf:.2f}")
    
    if 'dot_count_class' in detection_data:                    # ← CHANGE from 'nucleus_count_class'
        dot_idx = int(detection_data['dot_count_class'])
        dot_desc = morphology_mappings['dot_count_class'].get(dot_idx, 'unknown dot pattern')
        dot_conf = detection_data.get('dot_count_prob', 0.0)
        description_parts.append(f"and shows {dot_desc}")
        confidence_parts.append(f"dots: {dot_conf:.2f}")
    
    if 'flagellum_present_class' in detection_data:
        flagellum_idx = int(detection_data['flagellum_present_class'])
        flagellum_desc = morphology_mappings['flagellum_present_class'].get(flagellum_idx, 'unknown flagella status')
        flagellum_conf = detection_data.get('flagellum_present_prob', 0.0)
        description_parts.append(flagellum_desc)
        confidence_parts.append(f"flagella: {flagellum_conf:.2f}")
    
    if 'location_class' in detection_data:
        location_idx = int(detection_data['location_class'])
        location_desc = morphology_mappings['location_class'].get(location_idx, 'unknown location')
        location_conf = detection_data.get('location_prob', 0.0)
        description_parts.append(f"It is {location_desc}")
        confidence_parts.append(f"location: {location_conf:.2f}")
    
    # Join parts into natural sentence
    if len(description_parts) >= 2:
        description = f"{description_parts[0]} {description_parts[1]}"
        if len(description_parts) > 2:
            remaining = description_parts[2:]
            if len(remaining) == 1:
                description += f" and {remaining[0]}"
            else:
                description += f", {', '.join(remaining[:-1])}, and {remaining[-1]}"
    elif len(description_parts) == 1:
        description = description_parts[0]
    else:
        description = "Parasite detected with unknown morphological features"
    
    description += "."
    
    # Add confidence information
    confidence_text = f" (Confidence - {', '.join(confidence_parts)})"
    
    return description + confidence_text


def add_shape_info_to_bbox(bbox_res, shapes, bbox_num):
    """
    Add shape classification information to bbox results
    """
    
    if shapes is None:
        return bbox_res
        
    # Convert shapes to numpy if it's a tensor
    if hasattr(shapes, 'numpy'):
        shapes_np = shapes.numpy()
    else:
        shapes_np = shapes
        
    # Process shapes: [batch_size, num_queries, num_classes] -> shape predictions
    if len(shapes_np.shape) == 3:
        shapes_np = shapes_np[0]  # Remove batch dimension: [num_queries, num_classes]
    
    # Get shape probabilities and classes
    shape_probs = F.softmax(paddle.to_tensor(shapes_np), axis=-1).numpy()
    shape_classes = np.argmax(shape_probs, axis=-1)
    
    # Add shape info to each bbox detection
    for i, bbox_item in enumerate(bbox_res):
        if i < len(shape_classes):
            bbox_item['shape_class'] = int(shape_classes[i])
            bbox_item['shape_prob'] = float(shape_probs[i, shape_classes[i]])
    
    return bbox_res

def add_morphology_info_to_bbox(bbox_res, morph_data, morph_name):
    """
    Add morphological information to bbox results for any morphological feature
    """
    # print('iqra')
    if morph_data is None:
        return bbox_res
        
    # Convert to numpy if it's a tensor
    if hasattr(morph_data, 'numpy'):
        morph_np = morph_data.numpy()
    else:
        morph_np = morph_data
        
    # Handle batch processing - morph_np shape: [batch_size, num_predictions, num_classes]
    if len(morph_np.shape) == 3:
        batch_size, num_preds_per_image, num_classes = morph_np.shape
        
        # Get probabilities and classes for all images in batch
        morph_probs = F.softmax(paddle.to_tensor(morph_np), axis=-1).numpy()
        morph_classes = np.argmax(morph_probs, axis=-1)
        
        # Process each image in the batch
        pred_idx = 0
        for batch_idx in range(batch_size):
            # Get predictions for this batch image
            for pred_in_img in range(num_preds_per_image):
                if pred_idx < len(bbox_res):
                    bbox_res[pred_idx][f'{morph_name}_class'] = int(morph_classes[batch_idx, pred_in_img])
                    bbox_res[pred_idx][f'{morph_name}_prob'] = float(morph_probs[batch_idx, pred_in_img, morph_classes[batch_idx, pred_in_img]])
                pred_idx += 1
    else:
        # Handle non-batch case (original code)
        morph_probs = F.softmax(paddle.to_tensor(morph_np), axis=-1).numpy()
        morph_classes = np.argmax(morph_probs, axis=-1)
        
        for i, bbox_item in enumerate(bbox_res):
            if i < len(morph_classes):
                bbox_item[f'{morph_name}_class'] = int(morph_classes[i])
                bbox_item[f'{morph_name}_prob'] = float(morph_probs[i, morph_classes[i]])
    
    return bbox_res

def add_morphology_info_to_bbox1(bbox_res, morph_data, morph_name):
    """
    Add morphological information to bbox results for any morphological feature
    """

    if morph_data is None:
        return bbox_res
        
    # Convert to numpy if it's a tensor
    if hasattr(morph_data, 'numpy'):
        morph_np = morph_data.numpy()
    else:
        morph_np = morph_data
        
    if len(morph_np.shape) == 3:
        morph_np = morph_np[0]  
    
    # Get probabilities and classes
    morph_probs = F.softmax(paddle.to_tensor(morph_np), axis=-1).numpy()
    morph_classes = np.argmax(morph_probs, axis=-1)
    
    # Add morphological info to each bbox detection
    for i, bbox_item in enumerate(bbox_res):
        if i < len(morph_classes):
            bbox_item[f'{morph_name}_class'] = int(morph_classes[i])
            bbox_item[f'{morph_name}_prob'] = float(morph_probs[i, morph_classes[i]])
    
    return bbox_res


def get_infer_results(outs, catid, bias=0, save_threshold=0):
    """
    Get result at the stage of inference.
    The output format is dictionary containing bbox or mask result.

    For example, bbox result is a list and each element contains
    image_id, category_id, bbox and score.
    """
    # In get_infer_results(), add debugging before the morphological processing:
    # print(f"Debug - Processing batch with {len(im_id)} images")
    # print(f"Debug - im_id: {im_id}")
    print(f"Debug - 'shapes' in outs: {'shapes' in outs}")

    if 'shapes' in outs:
        print(f"Debug - outs['shapes'] type: {type(outs['shapes'])}")
        print(f"Debug - outs['shapes'] keys: {outs['shapes'].keys() if isinstance(outs['shapes'], dict) else 'not dict'}")
        
        # Check if shapes data has the right dimensions for the batch
        for morph_name, morph_data in outs['shapes'].items():
            print(f"Debug - {morph_name} shape: {morph_data.shape if hasattr(morph_data, 'shape') else 'no shape attr'}")
            
    if outs is None or len(outs) == 0:
        raise ValueError(
            'The number of valid detection result if zero. Please use reasonable model and check input data.'
        )

    im_id = outs['im_id']
    im_file = outs['im_file'] if 'im_file' in outs else None

    infer_res = {}
    if 'bbox' in outs:
        if len(outs['bbox']) > 0 and len(outs['bbox'][0]) > 6:
            infer_res['bbox'] = get_det_poly_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)
        else:
            infer_res['bbox'] = get_det_res(
                outs['bbox'],
                outs['bbox_num'],
                im_id,
                catid,
                bias=bias,
                im_file=im_file,
                save_threshold=save_threshold)

    if 'mask' in outs:
        # mask post process
        infer_res['mask'] = get_seg_res(outs['mask'], outs['bbox'],
                                        outs['bbox_num'], im_id, catid)

    if 'segm' in outs:
        infer_res['segm'] = get_solov2_segm_res(outs, im_id, catid)

    if 'keypoint' in outs:
        infer_res['keypoint'] = get_keypoint_res(outs, im_id)
        outs['bbox_num'] = [len(infer_res['keypoint'])]

    if 'pose3d' in outs:
        infer_res['pose3d'] = get_pose3d_res(outs, im_id)
        outs['bbox_num'] = [len(infer_res['pose3d'])]

    # Handle morphological features - process all 5 features
    if 'shapes' in outs and 'bbox' in infer_res and isinstance(outs['shapes'], dict):
        for morph_name, morph_data in outs['shapes'].items():
            infer_res['bbox'] = add_morphology_info_to_bbox(infer_res['bbox'], morph_data, morph_name)
        infer_res['shapes'] = outs['shapes']

    return infer_res



def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000),
                 classwise=False,
                 sigmas=None,
                 use_area=True):
    """
    Args:
        jsonfile (str): Evaluation json file, eg: bbox.json, mask.json.
        style (str): COCOeval style, can be `bbox` , `segm` , `proposal`, `keypoints` and `keypoints_crowd`.
        coco_gt (str): Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file (str): COCO annotations file.
        max_dets (tuple): COCO evaluation maxDets.
        classwise (bool): Whether per-category AP and draw P-R Curve or not.
        sigmas (nparray): keypoint labelling sigmas.
        use_area (bool): If gt annotations (eg. CrowdPose, AIC)
                         do not have 'area', please set use_area=False.
    """
    assert coco_gt != None or anno_file != None
    if style == 'keypoints_crowd':
        #please install xtcocotools==1.6
        from xtcocotools.coco import COCO
        from xtcocotools.cocoeval import COCOeval
    else:
        from pycocotools.coco import COCO
        try:
            from .fast_cocoeval import FastCOCOeval as COCOeval
        except:
            from pycocotools.cocoeval import COCOeval

    if coco_gt == None:
        coco_gt = COCO(anno_file)
    logger.info("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)
    if style == 'proposal':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    elif style == 'keypoints_crowd':
        coco_eval = COCOeval(coco_gt, coco_dt, style, sigmas, use_area)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if classwise:
        # Compute per-category AP and PR curve
        try:
            from terminaltables import AsciiTable
        except Exception as e:
            logger.error(
                'terminaltables not found, plaese install terminaltables. '
                'for example: `pip install terminaltables`.')
            raise e
        precisions = coco_eval.eval['precision']
        cat_ids = coco_gt.getCatIds()
        # precision: (iou, recall, cls, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]
        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = coco_gt.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (str(nm["name"]), '{:0.3f}'.format(float(ap))))
            pr_array = precisions[0, :, idx, 0, 2]
            recall_array = np.arange(0.0, 1.01, 0.01)
            draw_pr_curve(
                pr_array,
                recall_array,
                out_dir=style + '_pr_curve',
                file_name='{}_precision_recall_curve.jpg'.format(nm["name"]))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        logger.info('Per-category of {} AP: \n{}'.format(style, table.table))
        logger.info("per-category PR curve has output to {} folder.".format(
            style + '_pr_curve'))
    # flush coco evaluation result
    sys.stdout.flush()
    return coco_eval.stats


def json_eval_results(metric, json_directory, dataset):
    """
    cocoapi eval with already exists proposal.json, bbox.json or mask.json
    """
    assert metric == 'COCO'
    anno_file = dataset.get_anno()
    json_file_list = ['proposal.json', 'bbox.json', 'mask.json']
    if json_directory:
        assert os.path.exists(
            json_directory), "The json directory:{} does not exist".format(
                json_directory)
        for k, v in enumerate(json_file_list):
            json_file_list[k] = os.path.join(str(json_directory), v)

    coco_eval_style = ['proposal', 'bbox', 'segm']
    for i, v_json in enumerate(json_file_list):
        if os.path.exists(v_json):
            cocoapi_eval(v_json, coco_eval_style[i], anno_file=anno_file)
        else:
            logger.info("{} not exists!".format(v_json))
