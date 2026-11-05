# MorphXAI: An Explainable Framework for Morphological Analysis of Parasites in Blood Smear Images

Official implementation of our paper published at the **IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026**.

**Paper:** [MorphXAI: An Explainable Framework for Morphological Analysis of Parasites in Blood Smear Images](https://doi.org/10.1109/WACV61042.2026.00211)

**Authors:** Aqsa Yousaf, Sint Sint Win, Megan Coffee, and Habeeb Olufowobi.

## Overview

MorphXAI is an explainable deep learning framework that integrates parasite detection with fine-grained morphological analysis in blood smear images.

Built upon RT-DETRv3, MorphXAI introduces a morphological decoder that jointly predicts parasite bounding boxes, species, and five clinically relevant attributes: shape, curvature, visible dot count, flagellum presence, and developmental stage.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yousaf2026morphxai,
  title={MorphXAI: An Explainable Framework for Morphological Analysis of Parasites in Blood Smear Images},
  author={Yousaf, Aqsa and Win, Sint Sint and Coffee, Megan and Olufowobi, Habeeb},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026},
  doi={10.1109/WACV61042.2026.00211}
}
```

## Acknowledgments

This implementation builds upon [RT-DETRv3](https://github.com/clxia12/RT-DETRv3), developed by Shuo Wang, Chunlong Xia, Feng Lv, and Yifeng Shi. We acknowledge the original authors for making their implementation publicly available.

## License

This repository retains the Apache License 2.0 from the underlying RT-DETRv3 implementation. See [LICENSE](LICENSE) for details.
