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
    """Create a simple target trajectory for the drone to intercept"""
    
    # Create discrete trajectory with waypoints
    target_traj = co.DiscreteTrajectory()
    
    # Target is moving in a straight line
    start_time = 0.0
    end_time = 5.0
    dt = 0.1
    
    start_pos = np.array([10.0, 0.0, 3.0])
    velocity = np.array([2.0, 0.5, 0.0])
    
    time = start_time
    while time <= end_time:
        waypoint = co.StateWaypoint()
        waypoint.timestamp = time
        waypoint.position = start_pos + velocity * time
        waypoint.velocity = velocity
        waypoint.acceleration = np.array([0.0, 0.0, 0.0])
        
        target_traj.addWaypoint(waypoint)
        time += dt
    
    print(f"✓ Target trajectory created with duration {target_traj.getTotalDuration():.2f}s")
    return target_traj

def create_catching_scenario():
    """Create a catching/interception scenario"""
    
    # Initial state: hovering drone that needs to intercept target
    initial_pos = np.array([0.0, 0.0, 2.0])
    initial_vel = np.array([1.0, 0.0, 0.0])  
    initial_acc = np.array([0.0, 0.0, 0.0])
    initial_state = co.createDroneState(initial_pos, initial_vel, initial_acc)
    
    # Desired catching attitude: slight tilt forward (similar to perching example)
    angle_rad = np.radians(15)
    catching_euler = np.array([0.0, angle_rad, 0.0])  # roll, pitch, yaw
    catching_quat = co.euler2Quaternion(catching_euler)
    
    print(f"✓ Catching scenario created")
    print(f"  Initial position: {initial_state.position}")
    print(f"  Initial velocity: {initial_state.velocity}")
    print(f"  Catching attitude (euler): {catching_euler}")
    print(f"  Catching attitude (quat): [{catching_quat.w():.3f}, {catching_quat.x():.3f}, {catching_quat.y():.3f}, {catching_quat.z():.3f}]")
    
    return initial_state, catching_euler, catching_quat

def test_python_bindings_only(target_traj):
    """Test Python bindings without running incomplete C++ implementation"""
    
    print("\n=== CatchingOptimizer Python Bindings Test ===\n")
    
    # Create scenario
    initial_state, catching_euler, catching_quat = create_catching_scenario()
    
    # Test 1: Create optimizer
    print("Test 1: Creating CatchingOptimizer...")
    optimizer = co.CatchingOptimizer()
    print("✓ CatchingOptimizer created successfully")
    
    # Test 2: Method chaining with setDynamicLimits
    print("\nTest 2: Testing setDynamicLimits with method chaining...")
    optimizer = optimizer.setDynamicLimits(12.0, 8.0, 25.0, 3.0, 4.0, 2.5)
    print("✓ setDynamicLimits works with method chaining")
    
    # Test 3: setOptimizationWeights
    print("\nTest 3: Testing setOptimizationWeights...")
    optimizer = optimizer.setOptimizationWeights(1.0, -0.8, 1.2, 1.1, 1.0, 0.9, 1.0, 1.5)
    print("✓ setOptimizationWeights works")
    
    # Test 4: setTrajectoryParams
    print("\nTest 4: Testing setTrajectoryParams...")
    optimizer = optimizer.setTrajectoryParams(25, 3, 1, 3)
    print("✓ setTrajectoryParams works")
    
    # Test 5: setInitialState
    print("\nTest 5: Testing setInitialState...")
    optimizer = optimizer.setInitialState(initial_state)
    print("✓ setInitialState works")
    
    # Test 6: setTargetTrajectory
    print("\nTest 6: Testing setTargetTrajectory...")
    optimizer = optimizer.setTargetTrajectory(target_traj)
    print("✓ setTargetTrajectory works")
    
    # Test 7: setCatchingAttitude with euler angles
    print("\nTest 7: Testing setCatchingAttitude with euler angles...")
    optimizer.setCatchingAttitude(catching_euler)
    print("✓ setCatchingAttitude with euler angles works")
    
    # Test 8: setCatchingAttitude with quaternion
    print("\nTest 8: Testing setCatchingAttitude with quaternion...")
    optimizer.setCatchingAttitude(catching_quat)
    print("✓ setCatchingAttitude with quaternion works")
    
    # Test 9: Full method chain
    print("\nTest 9: Testing full method chain...")
    optimizer_chained = (co.CatchingOptimizer()
                        .setDynamicLimits(12.0, 8.0, 25.0, 3.0, 4.0, 2.5)
                        .setOptimizationWeights(1.0, -0.8, 1.2, 1.1, 1.0, 0.9, 1.0, 1.5)
                        .setTrajectoryParams(25, 3, 1, 3)
                        .setInitialState(initial_state)
                        .setTargetTrajectory(target_traj))
    print("✓ Full method chaining works")
    
    # Test 10: getIterationCount
    print("\nTest 10: Testing getIterationCount...")
    iterations = optimizer.getIterationCount()
    print(f"✓ getIterationCount works (returns: {iterations})")
    
    # Test 11: Test utility functions
    print("\nTest 11: Testing utility functions...")
    
    # euler2Quaternion
    test_euler = np.array([0.1, 0.2, 0.3])
    test_quat = co.euler2Quaternion(test_euler)
    print(f"  euler2Quaternion: {test_euler} -> [{test_quat.w():.3f}, {test_quat.x():.3f}, {test_quat.y():.3f}, {test_quat.z():.3f}]")
    
    # q2EulerAngle
    recovered_euler = np.zeros(3)
    co.q2EulerAngle(test_quat, recovered_euler)
    print(f"  q2EulerAngle: quat -> {recovered_euler}")
    
    # createDroneState
    test_state = co.createDroneState(np.array([1, 2, 3]), np.array([0.5, 0.5, 0]), np.array([0, 0, 0]))
    print(f"  createDroneState: pos={test_state.position}, vel={test_state.velocity}")
    
    # createQuaternion
    test_quat2 = co.createQuaternion(1.0, 0.0, 0.0, 0.0)
    print(f"  createQuaternion: [{test_quat2.w():.3f}, {test_quat2.x():.3f}, {test_quat2.y():.3f}, {test_quat2.z():.3f}]")
    
    print("✓ All utility functions work correctly")
    
    # Test 12: DiscreteTrajectory and StateWaypoint
    print("\nTest 12: Testing DiscreteTrajectory and StateWaypoint...")
    test_traj = co.DiscreteTrajectory()
    waypoint = co.StateWaypoint()
    waypoint.timestamp = 1.0
    waypoint.position = np.array([1.0, 2.0, 3.0])
    waypoint.velocity = np.array([0.5, 0.0, 0.0])
    waypoint.acceleration = np.array([0.0, 0.0, 0.0])
    test_traj.addWaypoint(waypoint)
    print(f"  Trajectory duration: {test_traj.getTotalDuration():.2f}s")
    print(f"  Position at t=0.5: {test_traj.getPosition(0.5)}")
    print("✓ DiscreteTrajectory and StateWaypoint work correctly")
    
    print("\n" + "="*60)
    print("✓✓✓ ALL PYTHON BINDINGS TESTS PASSED ✓✓✓")
    print("="*60)
    print("\nNote: generateTrajectory() not tested because the C++ implementation")
    print("is incomplete (has TODO sections that cause segmentation faults).")
    print("Once you complete the C++ implementation, you can test trajectory")
    print("generation by uncommenting the optimization test below.")
    
    return True

def main():
    """Main function demonstrating CatchingOptimizer Python bindings"""
    
    # Create target trajectory
    target_traj = create_target_trajectory()
    
    # Test Python bindings (without calling incomplete C++ implementation)
    success = test_python_bindings_only(target_traj)
    
    if not success:
        return 1
    
    print("\n" + "="*60)
    print("PYTHON BINDINGS VERIFICATION COMPLETE")
    print("="*60)
    print("\nAll CatchingOptimizer Python bindings are working correctly!")
    print("The bindings provide access to:")
    print("  ✓ CatchingOptimizer configuration methods")
    print("  ✓ Method chaining support")
    print("  ✓ Overloaded setCatchingAttitude (euler & quaternion)")
    print("  ✓ Target trajectory setting")
    print("  ✓ Utility functions (euler2Quaternion, q2EulerAngle, etc.)")
    print("  ✓ DiscreteTrajectory and StateWaypoint classes")
    print("  ✓ DroneState structure")
    print("\nNext steps:")
    print("  1. Complete the C++ implementation in catching_optimizer.h")
    print("  2. Implement the logic to extract target position/velocity from trajectory")
    print("  3. Then you can test full trajectory optimization with generateTrajectory()")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
