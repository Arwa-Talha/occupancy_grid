import sys
import glob
import numpy as np
import torch
import cv2
import time
import datetime
from tqdm import tqdm
import random

# Add CARLA Python API to sys.path
carla_egg_path = glob.glob(r'C:\CARLA\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg')
if not carla_egg_path:
    raise FileNotFoundError("CARLA .egg file not found. Check the path and CARLA installation.")
sys.path.append(carla_egg_path[0])
print(f"CARLA egg path added: {carla_egg_path[0]}")

import carla

from utils import (
    select_device, time_synchronized, non_max_suppression,
    scale_coords, plot_one_box, driving_area_mask, detected_lanes, show_seg_result, split_for_trace_model, letterbox
)

import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--model-path', type=str,default='D:/grad project/final_map/data/weights/yolopv2.pt')
    parser.add_argument('--save-dir', type=str, default='output')
    parser.add_argument('--img-size', type=int, default=352)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--fourcc', type=str, default='mp4v')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--frame-skip', type=int, default=2)
    return parser.parse_args()

def setup_carla_client(host, port, fps):  # Add fps as a parameter
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps  # Use the passed fps value
    world.apply_settings(settings)
    return world

def spawn_vehicle(world):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)
    print('✅ Ego vehicle spawned.')
    print('🚗 Autopilot enabled.')
    return vehicle

def spawn_traffic(world, num_vehicles=10, num_pedestrians=5):
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bps = blueprint_library.filter('vehicle.*')
    
    vehicles = []
    for i in range(num_vehicles):
        for attempt in range(3):  # Try 3 times
            spawn_point = random.choice(spawn_points)
            vehicle_bp = random.choice(vehicle_bps)
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                vehicle.set_autopilot(True)
                vehicles.append(vehicle)
                break  # Success
            except Exception as e:
                if attempt == 2:  # Last attempt failed
                    print(f"Failed to spawn traffic vehicle {i+1} after 3 attempts")
    return vehicles

def spawn_camera(world, vehicle, image_size=(1280, 720)):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(image_size[0]))
    camera_bp.set_attribute('image_size_y', str(image_size[1]))
    camera_bp.set_attribute('fov', '110')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    print(f'📷 Camera initialized at {time.time()}')
    return camera

def spawn_depth_camera(world, vehicle, image_size=(1280, 720)):
    blueprint_library = world.get_blueprint_library()
    depth_bp = blueprint_library.find('sensor.camera.depth')
    depth_bp.set_attribute('image_size_x', str(image_size[0]))
    depth_bp.set_attribute('image_size_y', str(image_size[1]))
    depth_bp.set_attribute('fov', '110')
    depth_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    depth_camera = world.spawn_actor(depth_bp, depth_transform, attach_to=vehicle)
    print(f'📷 Depth camera initialized at {time.time()}')
    return depth_camera

def load_model(model_path, device):
    try:
        model = torch.jit.load(model_path).to(device).eval()
        model = model.half()
        print(f"[DEBUG] Model loaded successfully on {device} with FP16")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

def depth_to_occupancy_grid(depth_image, grid_size=0.1, max_depth=50.0):
    try:
        # Debug input
        print(f"\n=== New Depth Frame ===")
        print(f"Original size: {depth_image.width}x{depth_image.height}")

        # Convert to numpy
        array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
        depth = array.reshape((depth_image.height, depth_image.width, 4))[:, :, :3]
        
        # Debug raw data
        print(f"Raw depth range - R: {depth[:,:,0].min()}-{depth[:,:,0].max()}, "
              f"G: {depth[:,:,1].min()}-{depth[:,:,1].max()}, "
              f"B: {depth[:,:,2].min()}-{depth[:,:,2].max()}")

        # Convert to meters
        depth_float = (depth[:, :, 0] + depth[:, :, 1]*256 + depth[:, :, 2]*65536) / (256**3 - 1)
        depth_float = depth_float * 1000  # Convert to meters

        # 3D projection
        fov = 110
        fx = depth_image.width / (2 * np.tan(np.deg2rad(fov)/2))
        u, v = np.meshgrid(np.arange(depth_image.width), np.arange(depth_image.height))
        
        x = (u - depth_image.width/2) * depth_float / fx
        z = depth_float
        
        # Ground removal
        ground_mask = (z > -0.5) & (z < 0.5)  # More lenient threshold
        ground_height = np.median(z[ground_mask]) if np.sum(ground_mask) > 100 else 0
        obstacle_mask = z > ground_height + 0.5  # 0.5m above ground
        
        print(f"Ground height: {ground_height:.2f}m, Obstacle points: {np.sum(obstacle_mask)}")

        # Create grid
        grid_x = np.clip((x[obstacle_mask] / grid_size).astype(int), 0, int(max_depth/grid_size)-1)
        grid_z = np.clip((z[obstacle_mask] / grid_size).astype(int), 0, int(max_depth/grid_size)-1)
        
        grid = np.zeros((int(max_depth/grid_size), int(max_depth/grid_size)), dtype=np.uint8)
        np.add.at(grid, (grid_z, grid_x), 1)
        
        # Debug output
        print(f"Occupancy grid: {grid.shape}, Obstacles: {np.sum(grid > 0)}")
        cv2.imwrite('debug_depth_projection.png', (grid * 255).astype(np.uint8))
        
        return (grid > 0).astype(np.uint8)
        
    except Exception as e:
        print(f"❌ Depth processing crashed: {str(e)}", exc_info=True)
        return np.zeros((int(max_depth/grid_size), int(max_depth/grid_size)), dtype=np.uint8)

def segmentation_to_bev(lanes, da_mask, depth_image=None, img_shape=(720, 1280), bev_shape=(500, 400), grid_size=0.1):
    fov = 110
    fx = img_shape[1] / (2 * np.tan(np.deg2rad(fov) / 2))
    fy = fx
    cx, cy = img_shape[1] / 2, img_shape[0] / 2

    h, w = img_shape
    if depth_image is not None:
        array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
        depth = array.reshape((h, w, 4))[:, :, :3]
        depth = depth[:, :, 0] * 0.003921568 + depth[:, :, 1] * 0.003921568 * 256 + depth[:, :, 2] * 0.003921568 * 65536
        depth = depth * 1000
    else:
        depth = np.full((h, w), 50.0)

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    bev_x = (z / grid_size).astype(int)
    bev_y = (x / grid_size).astype(int)
    
    bev_x += bev_shape[0] // 2
    bev_y += bev_shape[1] // 2

    bev_map = np.zeros(bev_shape, dtype=np.uint8)
    valid = (bev_x >= 0) & (bev_x < bev_shape[0]) & (bev_y >= 0) & (bev_y < bev_shape[1])
    
    lanes_valid = valid & (lanes == 1)
    bev_map[bev_x[lanes_valid], bev_y[lanes_valid]] = 1
    
    da_valid = valid & (da_mask == 1)
    bev_map[bev_x[da_valid], bev_y[da_valid]] = 2

    return bev_map

def combine_bev_map(bev_map, occupancy_grid):
    if occupancy_grid is None:
        return bev_map
    if bev_map.shape != occupancy_grid.shape:
        occupancy_grid = cv2.resize(occupancy_grid, (bev_map.shape[1], bev_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    bev_map[occupancy_grid == 1] = 3
    return bev_map

def visualize_bev_map(bev_map):
    color_map = np.zeros((bev_map.shape[0], bev_map.shape[1], 3), dtype=np.uint8)
    color_map[bev_map == 1] = [255, 255, 0]  # Yellow for lanes
    color_map[bev_map == 2] = [0, 255, 0]    # Green for drivable
    color_map[bev_map == 3] = [255, 0, 0]    # Red for obstacles
    cv2.imshow("BEV Map", color_map)
    return color_map
def process_image(image, writer, args, device, model, frame_count):
    try:
        frame_count[0] += 1  # Use a list to allow modification of frame_count
        if frame_count[0] % args.frame_skip != 0:
            return

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]
        img0 = array.copy()

        print(f"[DEBUG] letterbox function available: {'letterbox' in globals()}")
        t_pre = time_synchronized()
        img = letterbox(img0, args.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        # Preprocess image before model inference
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # Add batch dimension if needed
        img = img.half()  # Convert to half precision if using GPU
        t_pre_end = time_synchronized()
        print(f"[DEBUG] Preprocessing time for frame {frame_count[0]}: {t_pre_end - t_pre:.3f}s")

        t_inf = time_synchronized()
        with torch.no_grad():
            try:
                output = model(img)  # Model inference with preprocessed image
                print(f"[DEBUG] Raw output type: {type(output)}, length: {len(output)}")
                if len(output) == 3:
                    pred, seg, ll = output
                    print(f"[DEBUG] seg shape: {seg.shape}, ll shape: {ll.shape}")
                    # Check raw segmentation outputs
                    seg_max = torch.max(seg, dim=1)[0].cpu().numpy()
                    ll_max = torch.max(ll, dim=1)[0].cpu().numpy()
                    print(f"[DEBUG] seg max values: {np.unique(seg_max)}")
                    print(f"[DEBUG] ll max values: {np.unique(ll_max)}")
                    if isinstance(pred, tuple) and len(pred) == 2:
                        pred, anchor_grid = pred
                        pred = split_for_trace_model(pred, anchor_grid)
                    else:
                        pred = pred
                    pred = non_max_suppression(pred, 0.4, 0.5)
                else:
                    raise ValueError(f"Unexpected output length: {len(output)}")
                print(f"[DEBUG] Inference completed for frame {frame_count[0]}")
            except Exception as e:
                print(f"❌ Inference failed for frame {frame_count[0]}: {e}")
                raise
        t_inf_end = time_synchronized()
        print(f"[DEBUG] Inference time for frame {frame_count[0]}: {t_inf_end - t_inf:.3f}s")

        t_post = time_synchronized()
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    plot_one_box(xyxy, img0, label=f'{conf:.2f}', line_thickness=2)

        da_mask = driving_area_mask(seg)
        lanes = detected_lanes(ll)
        print(f"[DEBUG] da_mask unique values: {np.unique(da_mask)}")
        print(f"[DEBUG] lanes unique values: {np.unique(lanes)}")

        # Resize masks for overlay
        da_mask = cv2.resize(da_mask, (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_NEAREST)
        lanes = cv2.resize(lanes, (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_NEAREST)
        print(f"[DEBUG] Resized da_mask shape: {da_mask.shape}, unique: {np.unique(da_mask)}")
        print(f"[DEBUG] Resized lanes shape: {lanes.shape}, unique: {np.unique(lanes)}")

        # Save raw masks
        np.save(os.path.join(args.save_dir, f'da_mask_{frame_count[0]}.npy'), da_mask)
        np.save(os.path.join(args.save_dir, f'lanes_{frame_count[0]}.npy'), lanes)

        # Visualize segmentation
        show_seg_result(img0, (da_mask, lanes), is_demo=True)
        cv2.imwrite(os.path.join(args.save_dir, f'seg_frame_{frame_count[0]}.png'), img0)

        t_post_end = time_synchronized()
        fps = 1.0 / (t_post_end - t_pre + 1e-6)
        cv2.putText(img0, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        try:
            writer.write(cv2.resize(img0, (args.img_size, args.img_size)))
            print(f"📸 Frame {frame_count[0]} written successfully: FPS {fps:.2f}")
        except Exception as e:
            print(f"❌ Failed to write frame {frame_count[0]}: {e}")

        print(f"[DEBUG] Post-processing time for frame {frame_count[0]}: {t_post_end - t_post:.3f}s")
    except Exception as e:
        print(f"❌ Error processing frame {frame_count[0]}: {e}")


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print('[INFO] Connecting to CARLA...')
    world = setup_carla_client(args.host, args.port, args.fps)
    vehicle = None
    for actor in world.get_actors():
        if 'vehicle' in actor.type_id:
            vehicle = actor
            vehicle.set_autopilot(True)
            print('✅ Reused existing ego vehicle.')
            print('🚗 Autopilot enabled.')
            break
    if vehicle is None:
        vehicle = spawn_vehicle(world)

    spawn_traffic(world, num_vehicles=10)

    camera = spawn_camera(world, vehicle)
    # Comment out depth camera for now
    # depth_camera = spawn_depth_camera(world, vehicle)

    print('[INFO] Loading model...')
    print(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"[DEBUG] CUDA devices: {torch.cuda.device_count()}")
    device = select_device(args.device)
    if args.device != str(device) and 'cuda' in args.device:
        print(f"[WARNING] Requested device {args.device} unavailable, using {device} instead.")
    model = load_model(args.model_path, device)

    # Create dummy image with same shape as camera input (720p)
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img_lb, _, _ = letterbox(dummy_img, new_shape=args.img_size)
    img_lb = img_lb[:, :, ::-1].transpose(2, 0, 1)
    dummy_input = torch.from_numpy(img_lb.copy()).unsqueeze(0).to(device).float().div(255).half()

    print(f'🌡️ Starting model pre-warm-up at {time.time()}')
    try:
        with torch.no_grad():
            output = model(dummy_input)
            print(f"[DEBUG] Pre-warm-up output: {type(output)}, length: {len(output) if isinstance(output, (list, tuple)) else 'N/A'}")
            for i, item in enumerate(output):
                if hasattr(item, 'shape'):
                    print(f"[DEBUG] Output[{i}] shape: {item.shape}")
                    break
            else:
                print("[WARNING] No tensor with .shape found in output")
    except Exception as e:
        print(f"❌ Pre-warm-up failed: {e}")
        raise
    print(f'✅ Model pre-warm-up completed at {time.time()}')

    out_path = os.path.join(args.save_dir, f'output_{time.strftime("%Y-%m-%d_%H-%M-%S")}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (args.img_size, args.img_size))
    if not writer.isOpened():
        print(f"⚠️ Codec {args.fourcc} failed. Falling back to 'MJPG'.")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), args.fps, (args.img_size, args.img_size))
    print(f"[DEBUG] Video writer initialized for {out_path}")

    frame_count = 0
    fps_list = []

    print(f'✅ Starting frame capture and YOLOPv2 inference... at {time.time()}')
    camera.listen(process_image)

    try:
        while True:
            world.tick()
    except KeyboardInterrupt:
        print('🔻 Stopping...')
    except Exception as e:
        print(f"❌ Script crashed: {e}")
    finally:
        if writer is not None:
            writer.release()
            print(f"[DEBUG] Video writer released for {out_path}")
        camera.stop()
        if fps_list:
            avg_fps = sum(fps_list) / len(fps_list)
            print(f"🎯 Average FPS over {len(fps_list)} frames: {avg_fps:.2f}")
        print(f'🎬 Saved output video to {out_path}')

if __name__ == '__main__':
    main()