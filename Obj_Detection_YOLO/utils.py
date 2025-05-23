import torch
from collections import Counter




def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    '''
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint\corners, if boxes (x, y, w, h) or (x1, y1, x2, y2)

    Returns:
        tensor: Intersection over union for all examples
    '''


    # boxes_preds shape is (N, 4) where N is the number of boxes
    # boxes_labels shape is (N, 4)

    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4] # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]


    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    # box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)




# Non Max suppression

def non_max_suppression(bboxes, iou_threshold, box_format):
    """
    Performs Non-Max Suppression (NMS) on bounding boxes.

    Parameters:
        bboxes (list): List of bounding boxes with each box specified as 
                       [class, confidence, x1, y1, x2, y2] if box_format="corners".
        iou_threshold (float): IoU threshold for suppressing overlapping boxes.
        box_format (str): "corners" or "midpoint" to specify bounding box format.

    Returns:
        list: Remaining bounding boxes after applying NMS.
    """

    # bboxes is expected to be in the format of [[class, iou, x1, y1, x2, y2], [...], [...], ...]

    assert type(bboxes) == list
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    nms_bboxes = []

    while bboxes:
        chosen_bbox = bboxes.pop(0)
        bboxes = [box for box in bboxes if box[0] != chosen_bbox[0] or intersection_over_union(torch.tensor(box[2:]), torch.tensor(chosen_bbox[2:]), box_format=box_format) < iou_threshold]

        nms_bboxes.append(chosen_bbox)

    return sorted(nms_bboxes, key=lambda x:x[0])




# Mean Average Precision

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format='corners', num_classes=20):
    
    # pred_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...] 
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # say img 0 has 3 bboxes
        # and img 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5} 

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # amount bboxes = {0:torch.tensor([0, 0, 0]), 1:torch.tensor([0, 0, 0, 0, 0])}

        # this is for sorting over the probability score
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        total_true_boxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            
            else:
                FP[detection_idx] = 1

        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_boxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))
    
    return sum(average_precisions) / len(average_precisions)