#!/usr/bin/env python3
"""
Test script demonstrating the use of setInitialGuess for warm-starting optimization.
This provides a predicted intercept state to improve convergence and reduce iterations.
"""

import sys
import numpy as np

# Import the catching optimizer module
script_dir = Path(__file__).parent
project_root = script_dir.parent
build_dir = project_root / 'build'
from pathlib import Path

# Add build directory to path
sys.path.insert(0, str(build_dir))

import catching_optimizer_py as cop

def main():
    print("Testing CatchingOptimizer.setInitialGuess() method")
    print("=" * 60)
    
    # Create optimizer instance
    optimizer = cop.CatchingOptimizer()
    
    # Configure optimizer
    optimizer.setDynamicLimits(
        max_velocity=10.0,
        max_acceleration=10.0,
        max_thrust=20.0,
        min_thrust=2.0,
        max_body_rate=3.0,
        max_yaw_body_rate=2.0
    )
    
    optimizer.setOptimizationWeights(
        time_weight=1.0,
        position_weight=100.0,
        velocity_weight=10.0,
        acceleration_weight=1.0,
        thrust_weight=1.0,
        body_rate_weight=1.0,
        terminal_velocity_weight=-1.0,
        collision_weight=1.0
    )
    
    optimizer.setTrajectoryParams(
        integration_steps=20,
        traj_pieces_num=2,
        time_var_dim=1,
        custom_var_num=3
    )
    
    # Set initial state
    initial_pos = np.array([0.0, 0.0, 1.0])
    initial_vel = np.array([0.0, 0.0, 0.0])
    initial_acc = np.array([0.0, 0.0, 0.0])
    initial_state = cop.DroneState(initial_pos, initial_vel, initial_acc)
    optimizer.setInitialState(initial_state)
    
    # Set terminal state
    terminal_pos = np.array([5.0, 5.0, 2.0])
    terminal_vel = np.array([1.0, 0.0, 0.0])
    terminal_acc = np.array([0.0, 0.0, 0.0])
    terminal_state = cop.DroneState(terminal_pos, terminal_vel, terminal_acc)
    optimizer.setTerminalState(terminal_state)
    
    # Create a simple target trajectory
    target_traj = cop.DiscreteTrajectory()
    
    # Add waypoints to simulate a moving target
    for t in np.linspace(0, 3.0, 30):
        waypoint = cop.StateWaypoint()
        waypoint.timestamp = t
        waypoint.position = np.array([2.0 + t, 2.0 + 0.5*t, 1.5 + 0.2*t])
        waypoint.velocity = np.array([1.0, 0.5, 0.2])
        waypoint.acceleration = np.array([0.0, 0.0, 0.0])
        target_traj.addWaypoint(waypoint)
    
    optimizer.setTargetTrajectory(target_traj)
    
    # Set catching attitude (default upright)
    euler_attitude = np.array([0.0, 0.0, 0.0])
    optimizer.setCatchingAttitude(euler_attitude)
    
    # ============================================================
    # KEY FEATURE: Set initial guess for warm-starting
    # ============================================================
    # This provides a predicted intercept state to the optimizer
    # Based on some prediction algorithm (e.g., PN guidance, pure pursuit)
    
    predicted_intercept_time = 1.5  # seconds
    predicted_intercept_pos = np.array([3.5, 2.75, 1.8])
    predicted_intercept_vel = np.array([1.0, 0.5, 0.2])
    predicted_intercept_acc = np.array([0.0, 0.0, 0.0])
    
    print("\nSetting initial guess:")
    print(f"  Intercept time: {predicted_intercept_time:.2f} s")
    print(f"  Intercept position: {predicted_intercept_pos}")
    print(f"  Intercept velocity: {predicted_intercept_vel}")
    print(f"  Intercept acceleration: {predicted_intercept_acc}")
    
    optimizer.setInitialGuess(
        predicted_intercept_time,
        predicted_intercept_pos,
        predicted_intercept_vel,
        predicted_intercept_acc  # Optional, defaults to zero
    )
    
    # Generate trajectory
    print("\nGenerating trajectory with initial guess...")
    success, trajectory = optimizer.generateTrajectory()
    
    if success:
        print(f"✓ Trajectory generation successful!")
        print(f"  Total duration: {trajectory.getTotalDuration():.3f} s")
        print(f"  Number of pieces: {trajectory.getPieceNum()}")
        print(f"  Optimization iterations: {optimizer.getIterationCount()}")
        print(f"  Max velocity: {trajectory.getMaxVelRate():.3f} m/s")
        print(f"  Max acceleration: {trajectory.getMaxAccRate():.3f} m/s²")
        print(f"  Max thrust: {trajectory.getMaxThrust():.3f} N")
        
        # Sample the trajectory
        print("\nTrajectory samples:")
        print("  Time  |   Position   |   Velocity")
        print("-" * 50)
        for t in np.linspace(0, trajectory.getTotalDuration(), 5):
            pos = trajectory.getPos(t)
            vel = trajectory.getVel(t)
            print(f"  {t:4.2f}  | {pos[0]:5.2f},{pos[1]:5.2f},{pos[2]:5.2f} | {vel[0]:5.2f},{vel[1]:5.2f},{vel[2]:5.2f}")
    else:
        print("✗ Trajectory generation failed!")
        return 1
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
