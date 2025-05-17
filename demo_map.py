import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
from utils.utils import select_device, increment_path, LoadImages

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='D:/grad project/YOLOP 21/data/input/test_video.mp4', help='source')
    parser.add_argument('--weights', nargs='+', type=str, default=['data/weights/yolopv2.pt'], help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=384, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser

def detect(opt):
    source, weights, imgsz = opt.source, opt.weights, opt.img_size

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(opt.device)
    model = torch.jit.load(weights[0], map_location=device)
    half = device.type != 'cpu'
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    dataset = LoadImages(source, img_size=imgsz, stride=32)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    H = np.load('D:/grad project/YOLOP 21/homography.npy')

    h, w = 544, 960  # Match the actual video resolution
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        [pred, anchor_grid], seg, ll = model(img)
        print(f"seg shape: {seg.shape}, seg unique: {torch.unique(seg)}")
        print(f"ll shape: {ll.shape}, ll unique: {torch.unique(ll)}")

        # Segmentation map
        seg_map = torch.argmax(seg, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Lane line map (apply threshold)
        ll_map = (ll.squeeze(0).squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)

        # Apply homography to get BEV
        seg_map_bev = cv2.warpPerspective(seg_map, H, (w, h))
        ll_map_bev = cv2.warpPerspective(ll_map, H, (w, h))

        print(f"seg_map_bev unique values: {np.unique(seg_map_bev)}")
        print(f"ll_map_bev unique values: {np.unique(ll_map_bev)}")

        # Color lane BEV
        ll_map_bev_color = cv2.cvtColor(ll_map_bev * 255, cv2.COLOR_GRAY2BGR)
        seg_map_bev_color = cv2.applyColorMap(seg_map_bev * 127, cv2.COLORMAP_JET)

        # Save outputs
        filename = Path(path).stem
        cv2.imwrite(f'{save_dir}/bev_ll_{filename}_{dataset.frame}.png', ll_map_bev_color)
        cv2.imwrite(f'{save_dir}/bev_seg_{filename}_{dataset.frame}.png', seg_map_bev_color)
        cv2.imshow('Segmentation Map', seg_map * 100)  # Multiply to make classes visible

if __name__ == '__main__':
    opt = make_parser().parse_args()
    with torch.no_grad():
        detect(opt)
