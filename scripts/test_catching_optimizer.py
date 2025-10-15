#!/usr/bin/env python3
"""
Test example demonstrating CatchingOptimizer Python bindings
This example shows how to use the bindings for trajectory optimization,
following the pattern from advanced_python_example.py for comparison.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Import the catching optimizer module
script_dir = Path(__file__).parent
project_root = script_dir.parent
build_dir = project_root / 'build'

# Add build directory to path
sys.path.insert(0, str(build_dir))

try:
    import catching_optimizer_py as co
    print("✓ Module imported successfully")
    print(f"Version: {co.__version__}")
except ImportError as e:
    print(f"✗ Failed to import catching_optimizer_py: {e}")
    print(f"Make sure the module is built in: {build_dir}")
    sys.exit(1)

def create_target_trajectory():
    """Create a simple stationary target trajectory"""
    
    # Create discrete trajectory with waypoints
    target_traj = co.DiscreteTrajectory()
    
    # Target is stationary at a fixed position
    start_time = 0.0
    end_time = 15.0
    dt = 0.5
    
    # Stationary target position
    target_pos = np.array([5.0, 5.0, 1.5])
    velocity = np.array([0.0, 0.0, 0.0])
    
    time = start_time
    while time <= end_time:
        waypoint = co.StateWaypoint()
        waypoint.timestamp = time
        waypoint.position = target_pos
        waypoint.velocity = velocity
        waypoint.acceleration = np.array([0.0, 0.0, 0.0])
        
        target_traj.addWaypoint(waypoint)
        time += dt
    
    print(f"✓ Target trajectory created with duration {target_traj.getTotalDuration():.2f}s")
    return target_traj

def create_catching_scenario():
    """Create a catching/interception scenario"""
    
    # Initial state: hovering drone that needs to reach target
    initial_pos = np.array([0.0, 0.0, 1.0])
    initial_vel = np.array([0.0, 0.0, 0.0])  
    initial_acc = np.array([0.0, 0.0, 0.0])
    initial_state = co.createDroneState(initial_pos, initial_vel, initial_acc)
    
    # Desired catching attitude: slight tilt forward (similar to perching)
    angle_rad = np.radians(15)
    catching_euler = np.array([0.0, angle_rad, 0.0])  # roll, pitch, yaw
    catching_quat = co.euler2Quaternion(catching_euler)
    
    print(f"✓ Catching scenario created")
    print(f"  Initial position: {initial_state.position}")
    print(f"  Initial velocity: {initial_state.velocity}")
    print(f"  Catching attitude (euler): {catching_euler}")
    print(f"  Catching attitude (quat): [{catching_quat.w():.3f}, {catching_quat.x():.3f}, {catching_quat.y():.3f}, {catching_quat.z():.3f}]")
    
    return initial_state, catching_euler, catching_quat

def test_catching_optimizer(target_traj):
    """Test CatchingOptimizer with trajectory generation"""
    
    print("\n=== CatchingOptimizer Test ===\n")
    
    # Create scenario
    initial_state, catching_euler, catching_quat = create_catching_scenario()
    
    # Create target state to match the stationary target
    target_pos = np.array([5.0, 5.0, 1.5])
    target_vel = np.array([0.0, 0.0, 0.0])
    target_state = co.createDroneState(target_pos, target_vel, np.array([0.0, 0.0, 0.0]))
    
    print("Creating and configuring CatchingOptimizer...")
    optimizer = (co.CatchingOptimizer()
                .setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0)
                .setOptimizationWeights(1.0, 100.0, 10.0, 1.0, 1.0, 1.0, -1.0, 1.0)
                .setTrajectoryParams(20, 3, 1, 3)
                .setInitialState(initial_state)
                .setTargetTrajectory(target_traj)
                .setTerminalState(target_state))
    
    optimizer.setCatchingAttitude(catching_euler)
    
    print("✓ Optimizer configured")
    print("\nGenerating trajectory...")
    
    try:
        success, trajectory = optimizer.generateTrajectory()
        
        if success:
            print(f"✓ Trajectory generation SUCCESSFUL!")
            print(f"  Iterations: {optimizer.getIterationCount()}")
            print(f"  Duration: {trajectory.getTotalDuration():.3f}s")
            print(f"  Pieces: {trajectory.getPieceNum()}")
            
            # Get trajectory endpoints
            start_pos = trajectory.getJuncPos(0)
            end_pos = trajectory.getJuncPos(trajectory.getPieceNum())
            start_vel = trajectory.getJuncVel(0)
            end_vel = trajectory.getJuncVel(trajectory.getPieceNum())
            
            print(f"\n  Start: pos={start_pos}, vel={start_vel}")
            print(f"  End:   pos={end_pos}, vel={end_vel}")
            
            # Check constraints
            max_vel = trajectory.getMaxVelRate()
            max_acc = trajectory.getMaxAccRate()
            print(f"\n  Max velocity: {max_vel:.3f} m/s")
            print(f"  Max acceleration: {max_acc:.3f} m/s²")
            
            return True, trajectory
        else:
            print("✗ Trajectory generation FAILED")
            return False, None
            
    except Exception as e:
        print(f"✗ Exception during trajectory generation: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def save_trajectory_to_csv(trajectory, filename):
    """Save trajectory to CSV file for visualization"""
    
    # Sample trajectory at regular intervals
    dt = 0.01  # 100Hz sampling
    total_duration = trajectory.getTotalDuration()
    
    times = []
    positions = []
    velocities = []
    accelerations = []
    
    t = 0.0
    while t <= total_duration:
        times.append(t)
        positions.append(trajectory.getPos(t))
        velocities.append(trajectory.getVel(t))
        accelerations.append(trajectory.getAcc(t))
        t += dt
    
    # Convert to numpy arrays
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    
    # Save to CSV
    header = "time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z"
    data = np.column_stack([times, positions, velocities, accelerations])
    
    assets_dir = project_root / 'assets'
    assets_dir.mkdir(exist_ok=True)
    filepath = assets_dir / filename
    
    np.savetxt(filepath, data, delimiter=',', header=header, comments='')
    print(f"\n✓ Trajectory saved to {filepath}")
    return filepath


def main():
    """Main function demonstrating CatchingOptimizer"""
    
    # Create target trajectory
    target_traj = create_target_trajectory()
    
    # Test catching optimizer
    success, trajectory = test_catching_optimizer(target_traj)
    
    if not success:
        print("\n✗ Test FAILED")
        return 1
    
    # Save trajectory for visualization
    print("\nSaving trajectory...")
    save_trajectory_to_csv(trajectory, 'catching_optimizer_trajectory.csv')
    
    print("\n" + "="*60)
    print("✓✓✓ TEST COMPLETE ✓✓✓")
    print("="*60)
    print("\nCatchingOptimizer successfully generated a trajectory!")
    print("\nNext steps:")
    print("  1. Visualize trajectory: python scripts/visualize_trajectories.py")
    print("  2. View summary: python scripts/trajectory_summary.py")
    print("  3. Compare with perching: ./build/catching_perching_compare")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
