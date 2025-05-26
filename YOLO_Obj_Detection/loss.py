# the output of the YOLO v1 model is [c1, c2, .....cn, [Cp, x1, y1, w, h] * times number of bbox]
# so we take the difference of mean-squared value of the actual coordinates(x1, y1) from the predicted coordinates and do that for all the boudning bboxes. we do same for the height and wdith for all the bboxes. we also do this for the Cp(class probability) abd the classes(c1, c2, c3, ....cn) and add all of them to get the loss. this is the basic dumb down version of the loss funtion used in YOLO_v1

from utils import intersection_over_union

import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()

        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, best_box = torch.max(ious, dim=0)

        exists_box = target[..., 20].unsqueeze(dim=3) # Identity of object i
        # when we do `target[..., 20]` the last dimesion is lost so we add it back with `.unsqueeze(dim=3)`


        # ========================= #
        #    FOR BOX COORDINATES    #
        # ========================= #
        # for the coordintes (x1, y1) and the width and height (w and h) of the bbox and predictions

        # responsible for picking the box with the better IOU between the two (coordinates of the box only)
        box_predictions = exists_box * (
            best_box * (predictions[..., 26:30]) + (1 - best_box) * predictions[..., 21:25]
        )

        # the coordintes of the target / label
        box_targets = exists_box * target[..., 21:25]

        # square root of the height and width of the predictions
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))

        # square root of the height and width of the labels
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])


        # mean square value of all the 4 coordinates value are computed by 'self.mse()'
        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )


        # ===================== #
        #    FOR OBJECT LOSS    #
        # ===================== #
        # Penalizes the model for predicting wrong confiednce score in a grid with an image

        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # ======================== #
        #    FOR NO OBJECT LOSS    #
        # ======================== #
        # Penalizes the model for predicting wrong confidence score in a grid with no image
        # penalizes the model when it predicts high confidence scores for bounding boxes where no object exists.

        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )


        # ==================== #
        #    FOR CLASS LOSS    #
        # ==================== #

        # (N, S, S, 20) -> (N*S*S, 20) 
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )
        
        loss = (
            self.lambda_coord * box_loss # First two rows of loss in original YOLO_V1 paper
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        # return [box_loss, object_loss, no_object_loss, class_loss]
        return loss


loss_class = YoloLoss()


predictions = torch.randn(size=(32, 7, 7, 30))
target = torch.randn(size=(32, 7, 7, 25))


loss = loss_class(predictions, predictions)
print(loss)