import torch
import supervision as sv
from deva.ext.inference import Tracker


def get_detic_model():
    from detic import Detic
    detic_model = Detic(['cat'], masks=True).cuda()
    # detic_model = torch.hub.load('beasteers/Detic', 'detic')
    return detic_model


class DeticTracker(Tracker):
    def __init__(self, deva, detic):
        super().__init__(deva, 'cuda')
        self.detic = detic

    def predict_classes(self, image):
        prompt = self.cfg['prompt']
        if self.detic.metadata_name != prompt:
            self.detic.set_vocab(prompt.split('.'), metadata_name=prompt)
        outputs = self.detic(image)

        return sv.Detections(
            xyxy=outputs["instances"].pred_boxes.tensor.cpu().numpy(),
            mask=outputs["instances"].pred_masks.cpu().numpy() if hasattr(outputs["instances"], 'pred_masks') else None,
            confidence=outputs["instances"].scores.cpu().numpy(),
            class_id=outputs["instances"].pred_classes.cpu().numpy().astype(int),
        )
