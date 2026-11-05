# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import numpy as np
from ppdet.core.workspace import register, serializable

__all__ = ['GT2MorphologyTarget']


@register
@serializable
class GT2MorphologyTarget(object):
    """
    Convert ground truth morphological features to model targets.
    Maps string morphological features to integer class indices.
    """
    
    def __init__(self):
      
        self.morphology_mappings = {
            'shape_type': {
                'ring': 0,         # e.g. Babesia
                'elongated': 1,    # e.g. T. brucei, T. cruzi, Leishmania promastigote
                'amoeboid': 2,     # e.g. Leishmania amastigote
                'pyriform': 3,     # pear-shaped (Babesia variants)
                'crescent': 4,     # e.g. Toxoplasma (banana/crescent)
                'none': 5
            },
            'curvature': {
                'straight': 0,        # e.g. Leishmania
                'slightly_curved': 1, # e.g. T. brucei, Toxoplasma
                'C_or_S_shaped': 2,   # e.g. T. cruzi
                'none': 3             # round/amoeboid
            },
            'dot_count': {
                0: 0,   # no visible dots
                1: 1,   # one dot (e.g. nucleus only)
                2: 2,   # two dots (nucleus + kinetoplast, common in trypanosomes)
                3: 3    # more than 2 dots
            },
            'flagellum_present': {
                True: 1,
                False: 0
            },
            'location': {
                'intracellular': 0, # e.g. Babesia, Leishmania, Toxoplasma
                'extracellular': 1  # e.g. T. brucei, T. cruzi
            }
        }
    
    def __call__(self, sample, context):
        """
        Args:
            sample (dict): Sample data containing 'gt_morphology'
            context (dict): Context information
        
        Returns:
            sample (dict): Sample with processed morphological targets
        """
        if 'gt_morphology' not in sample:
            return sample
        
        gt_morphology = sample['gt_morphology']
        if gt_morphology is None or len(gt_morphology) == 0:
            return sample
        
        # Convert morphological features to integer targets
        processed_morphology = {}
        for feature_name, mapping in self.morphology_mappings.items():
            feature_targets = []
            
            for morph_data in gt_morphology:
                if feature_name in morph_data:
                    feature_value = morph_data[feature_name]
                    
                    # Handle different data types
                    if feature_name == 'dot_count':
                        # Handle integer values directly
                        target_idx = mapping.get(feature_value, 0)
                    elif feature_name == 'flagellum_present':
                        # Handle boolean values
                        target_idx = mapping.get(feature_value, 0)
                    else:
                        # Handle string values
                        target_idx = mapping.get(feature_value, 0)
                    
                    feature_targets.append(target_idx)
                else:
                    # Default value if feature is missing
                    feature_targets.append(0)
            
            # Convert to paddle tensor
            if len(feature_targets) > 0:
                processed_morphology[feature_name] = paddle.to_tensor(
                    feature_targets, dtype='int64')
            else:
                processed_morphology[feature_name] = paddle.to_tensor(
                    [], dtype='int64')
        
        sample['gt_morphology'] = processed_morphology
        return sample