# Mask DINO

The MMDetection implementation of [Mask DINO](https://arxiv.org/abs/2206.02777), which is taken from https://github.com/open-mmlab/mmdetection/pull/9808.

The evaluation reults of the pretrained weight are presented as follows.

```
03/20 19:28:09 - mmengine - INFO - Panoptic Evaluation Results:
+--------+--------+--------+--------+------------+
|        | PQ     | SQ     | RQ     | categories |
+--------+--------+--------+--------+------------+
| All    | 52.948 | 83.662 | 62.578 | 133        |
| Things | 58.958 | 84.918 | 69.075 | 80         |
| Stuff  | 43.878 | 81.767 | 52.771 | 53         |
+--------+--------+--------+--------+------------+
03/20 19:28:20 - mmengine - INFO - Evaluating bbox...
Evaluate annotation type *bbox*
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.489
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.686
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.532
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.320
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.520
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.641
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.680
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.811
03/20 19:29:00 - mmengine - INFO - bbox_mAP_copypaste: 0.489 0.686 0.532 0.320 0.520 0.641
03/20 19:29:00 - mmengine - INFO - Evaluating segm...
Evaluate annotation type *segm*
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.443
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.671
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.482
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.244
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.477
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.636
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.393
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.762
03/20 19:29:47 - mmengine - INFO - segm_mAP_copypaste: 0.443 0.671 0.482 0.244 0.477 0.636
03/20 19:29:48 - mmengine - INFO - Iter(test) [5000/5000]  coco_panoptic/PQ: 52.9484  coco_panoptic/SQ: 83.6622  coco_panoptic/RQ: 62.5779  coco_panoptic/PQ_th: 58.9575  coco_panoptic/SQ_th: 84.9180  coco_panoptic/RQ_th: 69.0747  coco_panoptic/PQ_st: 43.8780  coco_panoptic/SQ_st: 81.7665  coco_panoptic/RQ_st: 52.7714  coco/bbox_mAP: 0.4890  coco/bbox_mAP_50: 0.6860  coco/bbox_mAP_75: 0.5320  coco/bbox_mAP_s: 0.3200  coco/bbox_mAP_m: 0.5200  coco/bbox_mAP_l: 0.6410  coco/segm_mAP: 0.4430  coco/segm_mAP_50: 0.6710  coco/segm_mAP_75: 0.4820  coco/segm_mAP_s: 0.2440  coco/segm_mAP_m: 0.4770  coco/segm_mAP_l: 0.6360
```
