import carla
import numpy as np
import matplotlib.pyplot as plt
import heapq
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
def process_depth_image(image):
    # CARLA depth is encoded in RGB channels (0-1 normalized)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    depth = (array[:, :, :3].astype(np.float32) @ [1.0, 256.0, 65536.0]) / (256**3 - 1)
    depth *= 1000  # Convert to meters
    
    print(f"[DEBUG] Depth range: {depth.min():.2f}m - {depth.max():.2f}m")
    
    # Camera projection parameters
    fov = 90.0
    fx = (image.width / 2) / np.tan(np.radians(fov / 2))
    fy = fx
    cx, cy = image.width / 2, image.height / 2
    
    # Sample every 4th pixel for performance
    step = 4
    for i in range(0, image.height, step):
        for j in range(0, image.width, step):
            d = depth[i, j]
            if 0.5 < d < 50.0:  # Valid range
                # Convert to vehicle coordinates (X forward, Y left, Z up)
                x = d
                y = (j - cx) * d / fx
                z = (i - cy) * d / fy
                
                # Filter ground and noise
                if z > 0.3:  # 30cm above ground
                    x_idx, y_idx = world_to_grid(x, y)
                    if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                        occupancy_grid[x_idx, y_idx] = 1

# Improved LiDAR processing
def process_lidar_data(point_cloud):
    points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
    print(f"[DEBUG] LiDAR points: {len(points)}")
    
    for point in points:
        x, y, z = point[0], point[1], point[2]
        if z > 0.3:  # 30cm above ground
            x_idx, y_idx = world_to_grid(x, y)
            if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                occupancy_grid[x_idx, y_idx] = min(occupancy_grid[x_idx, y_idx] + 1, 3)  # Accumulate hits

# Enhanced visualization
def visualize_grid(grid, path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.T, origin='lower', cmap='hot', interpolation='nearest')
    if path:
        px, py = zip(*path)
        plt.plot(px, py, 'b-', linewidth=2, label='Path')
    plt.colorbar(label='Occupancy')
    plt.title("Enhanced Occupancy Grid")
    plt.xlabel("X (cells)")
    plt.ylabel("Y (cells)")
    plt.grid(True)
    plt.show()

# A* path planning algorithm with 8-directional movement
def a_star(start, goal, grid):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        x, y = current
        # 8-directional movement
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            neighbor = (x + dx, y + dy)
            if (0 <= neighbor[0] < grid.shape[0] and
                0 <= neighbor[1] < grid.shape[1] and
                grid[neighbor] == 0):
                cost = 1.414 if dx != 0 and dy != 0 else 1
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return None

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

# Setup depth camera
depth_bp = blueprint_library.find('sensor.camera.depth')
depth_bp.set_attribute('image_size_x', '128')
depth_bp.set_attribute('image_size_y', '128')
depth_bp.set_attribute('fov', '90')
depth_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
depth_cam = world.spawn_actor(depth_bp, depth_transform, attach_to=vehicle)
print("[INFO] Depth camera spawned")

# Sensor callbacks
def lidar_callback(data, grid_ref):
    process_lidar_data(data)

def depth_callback(image, grid_ref):
    process_depth_image(image)

# Start listening to sensor data
lidar.listen(lambda data: lidar_callback(data, occupancy_grid))
depth_cam.listen(lambda image: depth_callback(image, occupancy_grid))

print("جمع البيانات لمدة 30 ثانية...")
time.sleep(30)  # Collect data for 30 seconds

# Stop sensors
lidar.stop()
depth_cam.stop()

# Debug occupancy grid
print(f"[DEBUG] Occupancy grid non-zero count: {np.sum(occupancy_grid > 0)}")
print(f"[DEBUG] Max occupancy value: {np.max(occupancy_grid)}")

# Plan path using A*
start = (50, 50)  # Near vehicle starting position
goal = (70, 70)   # Nearby goal
print(f"[DEBUG] Start cell ({start}) occupied: {occupancy_grid[start] > 0}")
print(f"[DEBUG] Goal cell ({goal}) occupied: {occupancy_grid[goal] > 0}")
path = a_star(start, goal, occupancy_grid)

# Visualize the grid and path
visualize_grid(occupancy_grid, path)

# Cleanup
vehicle.set_autopilot(False)
vehicle.destroy()
lidar.destroy()
depth_cam.destroy()
print("[INFO] Cleaned up actors")