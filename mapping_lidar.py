import carla
import numpy as np
import matplotlib.pyplot as plt
import weakref
import time

# Test connection to CARLA
try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print("✅ Successfully connected to CARLA world:", world.get_map().name)
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)

# Setup the occupancy grid
grid_width = 100
grid_height = 100
cell_size = 1.0  # meter
occupancy_grid = np.zeros((grid_width, grid_height), dtype=np.uint8)

# Convert real-world coordinates to grid coordinates
def world_to_grid(x, y):
    x_idx = int(x // cell_size) + grid_width // 2
    y_idx = int(y // cell_size) + grid_height // 2
    return x_idx, y_idx

# Improved depth processing
def process_depth_image(image, frame_times):
    start_time = time.time()
    # CARLA depth is encoded in RGB channels (0-1 normalized)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    depth = (array[:, :, :3].astype(np.float32) @ [1.0, 256.0, 65536.0]) / (256**3 - 1)
    depth *= 1000  # Convert to meters
    depth = np.clip(depth, 0.5, 50.0)  # Clip to valid range
    print(f"[DEBUG] Depth range: {depth.min():.2f}m - {depth.max():.2f}m")
    
    # Camera projection parameters
    fov = 90.0
    fx = (image.width / 2) / np.tan(np.radians(fov / 2))
    fy = fx
    cx, cy = image.width / 2, image.height / 2
    
    # Sample every 2nd pixel for better detail
    step = 2
    for i in range(0, image.height, step):
        for j in range(0, image.width, step):
            d = depth[i, j]
            if 0.5 < d < 50.0:  # Valid range (already clipped, but for clarity)
                # Convert to vehicle coordinates (X forward, Y left, Z up)
                x = d
                y = (j - cx) * d / fx
                z = (i - cy) * d / fy
                
                # Filter ground and noise
                if z > 0.3:  # 30cm above ground
                    x_idx, y_idx = world_to_grid(x, y)
                    if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                        occupancy_grid[x_idx, y_idx] = 1
    
    frame_times.append(time.time() - start_time)

# Improved LiDAR processing
def process_lidar_data(point_cloud, frame_times):
    start_time = time.time()
    points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
    print(f"[DEBUG] LiDAR points: {len(points)}")
    
    for point in points:
        x, y, z = point[0], point[1], point[2]
        if z > 0.3:  # 30cm above ground
            x_idx, y_idx = world_to_grid(x, y)
            if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                occupancy_grid[x_idx, y_idx] = min(occupancy_grid[x_idx, y_idx] + 1, 2)  # Cap at 2
    
    frame_times.append(time.time() - start_time)

# Enhanced visualization (without path)
def visualize_grid(grid, fps=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.T, origin='lower', cmap='hot', interpolation='nearest')
    if fps:
        plt.text(5, 5, f'Avg FPS: {fps:.1f}', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
    plt.colorbar(label='Occupancy')
    plt.title("Occupancy Grid")
    plt.xlabel("X (cells)")
    plt.ylabel("Y (cells)")
    plt.grid(True)
    plt.show()

# Setup vehicle and sensors
blueprint_library = world.get_blueprint_library()
print("Available vehicle blueprints:")
for bp in blueprint_library.filter('vehicle.*'):
    print(bp.id)

# Select vehicle blueprint with fallback
vehicle_bps = blueprint_library.filter('vehicle.tesla.model3')
if not vehicle_bps:
    print("❌ No 'vehicle.tesla.model3' found. Using first available vehicle instead.")
    vehicle_bps = blueprint_library.filter('vehicle.*')
    if not vehicle_bps:
        raise RuntimeError("❌ No vehicles found in blueprint library.")
vehicle_bp = vehicle_bps[0]
print(f"[INFO] Using vehicle blueprint: {vehicle_bp.id}")

# Spawn the vehicle
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)
print(f"[INFO] Vehicle spawned at: {spawn_point.location}")
print("[INFO] Autopilot enabled for vehicle")

# Setup LiDAR sensor
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '50')
lidar_transform = carla.Transform(carla.Location(x=0, z=2))
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
print("[INFO] LiDAR sensor spawned")

# Setup depth camera with higher resolution
depth_bp = blueprint_library.find('sensor.camera.depth')
depth_bp.set_attribute('image_size_x', '256')
depth_bp.set_attribute('image_size_y', '256')
depth_bp.set_attribute('fov', '90')
depth_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
depth_cam = world.spawn_actor(depth_bp, depth_transform, attach_to=vehicle)
print("[INFO] Depth camera spawned")

# Lists to track frame times for FPS
lidar_frame_times = []
depth_frame_times = []

# Sensor callbacks
def lidar_callback(data, grid_ref):
    process_lidar_data(data, lidar_frame_times)

def depth_callback(image, grid_ref):
    process_depth_image(image, depth_frame_times)

# Start listening to sensor data
lidar.listen(lambda data: lidar_callback(data, occupancy_grid))
depth_cam.listen(lambda image: depth_callback(image, occupancy_grid))

print("collecting data..")
time.sleep(30)  # Collect data for 30 seconds

# Stop sensors
lidar.stop()
depth_cam.stop()

# Calculate average FPS
lidar_fps = len(lidar_frame_times) / sum(lidar_frame_times) if lidar_frame_times else 0
depth_fps = len(depth_frame_times) / sum(depth_frame_times) if depth_frame_times else 0
avg_fps = (lidar_fps + depth_fps) / 2 if (lidar_fps and depth_fps) else max(lidar_fps, depth_fps)
print(f"[DEBUG] LiDAR FPS: {lidar_fps:.1f}")
print(f"[DEBUG] Depth FPS: {depth_fps:.1f}")
print(f"[DEBUG] Average FPS: {avg_fps:.1f}")

# Debug occupancy grid
print(f"[DEBUG] Occupancy grid non-zero count: {np.sum(occupancy_grid > 0)}")
print(f"[DEBUG] Max occupancy value: {np.max(occupancy_grid)}")

# Visualize the grid
visualize_grid(occupancy_grid, avg_fps)

# Cleanup
vehicle.set_autopilot(False)
vehicle.destroy()
lidar.destroy()
depth_cam.destroy()
print("[INFO] Cleaned up actors")