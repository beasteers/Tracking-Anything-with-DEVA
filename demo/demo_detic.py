import os
from os import path
from argparse import ArgumentParser

import cv2
import torch
import numpy as np
import supervision as sv
from tqdm import tqdm

from deva.inference.inference_core import DEVAInferenceCore
# from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
# from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args, add_auto_default_args
# from deva.ext.grounding_dino import get_grounding_dino_model
# from deva.ext.with_text_processor import process_frame_with_text as process_frame

from deva.ext.detic import get_detic_model, DeticTracker
from deva.ext.inference import Visualizer

from pyinstrument import Profiler


def main():
    torch.autograd.set_grad_enabled(False)

    # for id2rgb
    np.random.seed(42)
    """
    Arguments loading
    """
    parser = ArgumentParser()

    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)
    # add_auto_default_args(parser)
    deva_model, cfg, args = get_model_and_config(parser)
    # gd_model, sam_model = get_grounding_dino_model(cfg, 'cuda')
    """
    Temporal setting
    """
    cfg['temporal_setting'] = args.temporal_setting.lower()
    assert cfg['temporal_setting'] in ['semionline', 'online']
    cfg['pluralize'] = not args.do_not_pluralize

    # get data
    out_path = cfg['output'] or 'diva_output.mp4'

    print('Configuration:', cfg)

    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = cfg['num_voting_frames'] - 1
    deva.enabled_long_id()

    detic_model = get_detic_model()
    detic_model.set_vocab(cfg['prompt'].split('.'), metadata_name=cfg['prompt'])

    tracker = DeticTracker(deva, detic_model)
    vis = Visualizer(cfg, object_manager=deva.object_manager)

    video_info, WH = get_video_info(cfg['img_path'], 720)

    ti = 1
    prof = Profiler()
    try:
        with sv.VideoSink(out_path, video_info=video_info) as s, prof:
            pbar = tqdm(enumerate(sv.get_video_frames_generator(cfg['img_path'])), total=video_info.total_frames)
            for i, frame in pbar:
                frame = cv2.resize(frame, WH)
                # detections = tracker.predict_classes(frame)
                # out_frame = vis.draw_detections(frame.copy(), detections, np.array(CLASSES)[detections.class_id])
                # s.write_frame(out_frame)
                results = tracker(frame, ti)
                for (p, fi) in results:
                    torch.cuda.synchronize()
                    detections, labels = vis.as_detections(p, fi)
                    out_frame = vis.draw_detections(frame.copy(), detections, labels)
                    s.write_frame(out_frame)
                ti += 1
    finally:
        prof.print()


def get_video_info(src, size, fps_down=1, nrows=1, ncols=1):
    # get the source video info
    video_info = sv.VideoInfo.from_video_path(video_path=src)
    # make the video size a multiple of 16 (because otherwise it won't generate masks of the right size)
    aspect = video_info.width / video_info.height
    video_info.width = int(aspect*size)//16*16
    video_info.height = int(size)//16*16
    WH = video_info.width, video_info.height

    # double width because we have both detic and xmem frames
    video_info.width *= ncols
    video_info.height *= nrows
    # possibly reduce the video frame rate
    video_info.og_fps = video_info.fps
    video_info.fps /= fps_down or 1

    print(f"Input Video {src}\nsize: {WH}  fps: {video_info.fps}")
    return video_info, WH


if __name__ == '__main__':
    main()























# import os
# from os import path
# from argparse import ArgumentParser

# import torch

# import numpy as np

# from deva.model.network import DEVA
# from deva.inference.eval_args import add_common_eval_args
# from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
# from deva.inference.inference_core import DEVAInferenceCore
# from deva.ext.detic import get_detic_model, DeticTracker
# from deva.ext.inference import Visualizer

# from tqdm import tqdm
# import json

# import supervision as sv
# from pyinstrument import Profiler


# @torch.no_grad()
# def main(src, vocab):
#     # for id2rgb
#     np.random.seed(42)

#     # default parameters
#     parser = ArgumentParser()
#     add_common_eval_args(parser)
#     add_ext_eval_args(parser)
#     add_text_default_args(parser)

#     # load model and config
#     args = parser.parse_args([])
#     cfg = vars(args)
    

#     torch.autograd.set_grad_enabled(False)

#     # for id2rgb
#     np.random.seed(42)
#     """
#     Arguments loading
#     """
#     parser = ArgumentParser()

#     add_common_eval_args(parser)
#     add_ext_eval_args(parser)
#     add_auto_default_args(parser)
#     deva_model, cfg, args = get_model_and_config(parser)

#     cfg['enable_long_term_count_usage'] = True
#     cfg['max_num_objects'] = 50
#     cfg['size'] = 480
#     cfg['DINO_THRESHOLD'] = 0.35
#     cfg['amp'] = True
#     cfg['chunk_size'] = 4
#     cfg['detection_every'] = 5
#     cfg['max_missed_detection_count'] = 10
#     cfg['temporal_setting'] = 'online' # semionline usually works better; but online is faster for this demo
#     cfg['pluralize'] = True

#     detic_model = get_detic_model()

#     cfg['prompt'] = '.'.join(vocab)
#     detic_model.set_labels(cfg['prompt'].split('.'), metadata_name=cfg['prompt'])

#     deva = DEVAInferenceCore(deva_model, config=cfg)
#     deva.next_voting_frame = cfg['num_voting_frames'] - 1
#     deva.enabled_long_id()

#     # tracker = DINOTracker(deva, gd_model, fsam_model)
#     tracker = DeticTracker(deva, detic_model)
#     vis = Visualizer(cfg, object_manager=deva.object_manager)

#     video_info, WH = get_video_info(SOURCE_VIDEO_PATH, 240)

#     ti = 1
#     prof = Profiler()
#     try:
#         with sv.VideoSink(OUTPUT_VIDEO_PATH, video_info=video_info) as s, prof:
#             pbar = tqdm(enumerate(sv.get_video_frames_generator(SOURCE_VIDEO_PATH)), total=video_info.total_frames)
#             for i, frame in pbar:
#                 frame = cv2.resize(frame, WH)
#                 # detections = tracker.predict_classes(frame)
#                 # out_frame = vis.draw_detections(frame.copy(), detections, np.array(CLASSES)[detections.class_id])
#                 # s.write_frame(out_frame)
#                 results = tracker(frame, ti)
#                 for (p, fi) in results:
#                     torch.cuda.synchronize()
#                     detections, labels = vis.as_detections(p, fi)
#                     out_frame = vis.draw_detections(frame.copy(), detections, labels)
#                     s.write_frame(out_frame)
#                 ti += 1
#     finally:
#         prof.print()

# if __name__ == '__main__':
#     import fire
#     fire.Fire(main)