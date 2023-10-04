import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import supervision as sv

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.object_info import ObjectInfo
from deva.inference.frame_utils import FrameInfo
from deva.inference.demo_utils import get_input_frame_for_deva


class Tracker:
    def __init__(self, deva: DEVAInferenceCore, device):
        self.deva = deva
        self.cfg = deva.config
        self.device = device

    @torch.inference_mode()
    def __call__(self, image_np, ti):
        image_np = np.ascontiguousarray(image_np[:, :, ::-1])  # bgr2rgb
        image = get_input_frame_for_deva(image_np, self.cfg['size'])
        frame_info = FrameInfo(image, None, None, ti, {'shape': image_np.shape[:2]})

        if self.cfg['temporal_setting'] == 'semionline':
            return self.predict_semionline(image, image_np, frame_info, ti)
        elif self.cfg['temporal_setting'] == 'online':
            return self.predict_online(image, image_np, frame_info, ti)
        raise ValueError(f"Unknown temporal setting: {self.cfg['temporal_setting']}")

    @torch.inference_mode()
    def flush(self):
        results = []
        for frame_info in self.deva.frame_buffer:
            prob = self.deva.step(frame_info.image, None, None)
            results.append((prob, None, frame_info))
        return results

    def predict_classes(self, image):
        raise NotImplemented

    def predict_online(self, image, image_np, frame_info, ti):
        if ti % self.cfg['detection_every'] == 0:
            # incorporate new detections
            print(f"Detecting {ti} % {self.cfg['detection_every']}")
            detections = self.predict_classes(image_np)
            mask, segments_info = merge_masks(detections, image_np.shape, self.cfg['size'], self.device)
            frame_info.segments_info = segments_info
            # frame_info.mask = mask

            prob = self.deva.incorporate_detection(image, mask, segments_info)
            return [(prob, frame_info)]

        # Run the model on this frame
        prob = self.deva.step(image, None, None)
        return [(prob, frame_info)]

    def predict_semionline(self, image, image_np, frame_info, ti):
        results = []
        if ti + self.cfg['num_voting_frames'] > self.deva.next_voting_frame:
            print(f"Begin voting {ti} .. {self.deva.next_voting_frame}")
            detections = self.predict_classes(image_np)
            mask, segments_info = merge_masks(detections, image.shape, self.cfg['size'], self.device)
            frame_info.segments_info = segments_info
            frame_info.mask = mask

            # wait for more frames before proceeding
            self.deva.add_to_temporary_buffer(frame_info)

            if ti == self.deva.next_voting_frame:
                print(f"Finished voting. Predicting {ti} .. {self.deva.next_voting_frame}")
                res = self.predict_vote_in_buffer()
                results.extend(res)
        else:
            # standard propagation
            prob = self.deva.step(image, None, None)
            results.append((prob, frame_info))
        return results

    def predict_vote_in_buffer(self):
        results = []

        # process this clip
        frame_info = self.deva.frame_buffer[0]
        _, mask, segments_info = self.deva.vote_in_temporary_buffer(keyframe_selection='first')
        frame_info.segments_info = segments_info
        # frame_info.mask = mask

        prob = self.deva.incorporate_detection(frame_info.image, mask, segments_info)
        self.deva.next_voting_frame += self.cfg['detection_every']
        results.append((prob, frame_info))

        # step through the next frames
        for frame_info in self.deva.frame_buffer[1:]:
            prob = self.deva.step(frame_info.image, None, None)
            results.append((prob, frame_info))

        self.deva.clear_buffer()
        return results


def merge_masks(detections, image_shape, min_side, device):
    h, w = image_shape[:2]
    if min_side > 0:
        scale = min_side / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = h, w

    output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=device)
    curr_id = 1
    segments_info = []

    # sort by descending area to preserve the smallest object
    for i in np.flip(np.argsort(detections.area)):
        mask = detections.mask[i]
        confidence = detections.confidence[i]
        class_id = detections.class_id[i]
        mask = torch.from_numpy(mask.astype(np.float32))
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (new_h, new_w), mode='bilinear')[0, 0]
        mask = (mask > 0.5).float()

        if mask.sum() > 0:
            output_mask[mask > 0] = curr_id
            segments_info.append(ObjectInfo(id=curr_id, category_id=class_id, score=confidence))
            curr_id += 1

    return output_mask, segments_info




class Visualizer:
    need_remapping=True
    def __init__(self, cfg, object_manager) -> None:
        self.cfg = cfg
        self.object_manager = object_manager
        self.ba = sv.BoxAnnotator()
        self.ma = sv.MaskAnnotator()

    def as_detections(self, prob, frame_info):
        shape = frame_info.shape
        # need_resize = frame_info.need_resize
        prompts = self.cfg['prompt'].split('.')
        if prob.shape[-2] != shape:
            prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]
        # Probability mask -> index mask
        mask = torch.argmax(prob, dim=0)
        # remap indices
        if self.need_remapping:
            mask = self.object_manager.tmp_to_obj_cls(mask)

        # record output in the json file
        all_segments_info = self.object_manager.get_current_segments_info()

        # draw bounding boxes for the prompts
        all_masks = []
        labels = []
        all_cat_ids = []
        all_scores = []
        for seg in all_segments_info:
            m = mask == seg['id']
            if not m.any():
                continue
            all_masks.append(m)
            labels.append(f'{prompts[seg["category_id"]]} {seg["score"]:.2f}')
            all_cat_ids.append(seg['category_id'])
            all_scores.append(seg['score'])

        if all_masks:
            all_masks = torch.stack(all_masks, dim=0)
            xyxy = torchvision.ops.masks_to_boxes(all_masks)
        else:
            all_masks = torch.zeros((0, *mask.shape))
            xyxy = torch.zeros((0, 4))
        detections = sv.Detections(
            xyxy=xyxy.cpu().numpy(),
            mask=all_masks.cpu().numpy(),
            confidence=np.array(all_scores),
            class_id=np.array(all_cat_ids))
        labels = np.array(labels, dtype=object)
        return detections, labels

    def draw_detections(self, frame, detections, labels):
        frame = self.ba.annotate(frame, detections, labels=labels)
        frame = self.ma.annotate(frame, detections)
        return frame