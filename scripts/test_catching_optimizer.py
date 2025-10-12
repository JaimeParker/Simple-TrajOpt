#!/usr/bin/env python3
"""
Example script demonstrating the use of SimpleCatching optimizer for target interception.
This script shows how to:
1. Define a target trajectory
2. Configure the catching optimizer
3. Generate an interception trajectory
"""

import sys
sys.path.append('build')

import numpy as np
import catching_optimizer_py as catching


def create_stationary_target_trajectory(target_pos, duration, num_waypoints=50):
    """
    Create a DiscreteTrajectory for a stationary target (fixed point).
    
    Args:
        target_pos: Fixed position [x, y, z]
        duration: Total duration of the trajectory
        num_waypoints: Number of waypoints to generate
    
    Returns:
        DiscreteTrajectory object populated with waypoints
    """
    trajectory = catching.DiscreteTrajectory()
    
    target_pos = np.array(target_pos)
    velocity = np.array([0.0, 0.0, 0.0])  # Zero velocity for stationary target
    acceleration = np.array([0.0, 0.0, 0.0])  # Zero acceleration
    
    for i in range(num_waypoints):
        t = i * duration / (num_waypoints - 1)
        
        # Create a waypoint
        waypoint = catching.StateWaypoint()
        waypoint.timestamp = t
        waypoint.position = target_pos  # Fixed position
        waypoint.velocity = velocity
        waypoint.acceleration = acceleration
        
        # Add to trajectory
        trajectory.addWaypoint(waypoint)
    
    return trajectory


def create_linear_target_trajectory(initial_pos, velocity, duration, num_waypoints=50):
    """
    Create a DiscreteTrajectory for a linearly moving target.
    (Kept for future testing with moving targets)
    
    Args:
        initial_pos: Initial position [x, y, z]
        velocity: Constant velocity [vx, vy, vz]
        duration: Total duration of the trajectory
        num_waypoints: Number of waypoints to generate
    
    Returns:
        DiscreteTrajectory object populated with waypoints
    """
    trajectory = catching.DiscreteTrajectory()
    
    initial_pos = np.array(initial_pos)
    velocity = np.array(velocity)
    acceleration = np.array([0.0, 0.0, 0.0])  # Zero acceleration for linear motion
    
    for i in range(num_waypoints):
        t = i * duration / (num_waypoints - 1)
        
        # Create a waypoint
        waypoint = catching.StateWaypoint()
        waypoint.timestamp = t
        waypoint.position = initial_pos + velocity * t
        waypoint.velocity = velocity
        waypoint.acceleration = acceleration
        
        # Add to trajectory
        trajectory.addWaypoint(waypoint)
    
    return trajectory


def main():
    print("=" * 60)
    print("SimpleCatching Optimizer Example")
    print("=" * 60)
    
    # ---- Configuration ----
    # Initial drone state
    initial_pos = np.array([0.0, 0.0, 1.0])
    initial_vel = np.array([0.0, 0.0, 0.0])
    initial_acc = np.array([0.0, 0.0, 0.0])
    
    initial_state = catching.createDroneState(initial_pos, initial_vel, initial_acc)
    
    print(f"\nInitial Drone State:")
    print(f"  Position: {initial_state.position}")
    print(f"  Velocity: {initial_state.velocity}")
    print(f"  Acceleration: {initial_state.acceleration}")
    
    # Target trajectory (stationary target - fixed point)
    target_position = [5.0, 5.0, 1.5]
    
    # Create target trajectory using DiscreteTrajectory with waypoints
    # Use a reasonable duration for the drone to reach the target
    estimated_distance = np.linalg.norm(np.array(target_position) - initial_pos)
    estimated_duration = estimated_distance / 3.0  # Assume average speed of 3 m/s
    estimated_duration = max(5.0, estimated_duration)  # At least 5 seconds for safety
    
    target_traj = create_stationary_target_trajectory(
        target_position, 
        estimated_duration,
        num_waypoints=50
    )
    
    print(f"\nTarget Trajectory (Stationary):")
    print(f"  Position: {target_position}")
    print(f"  Velocity: [0.0, 0.0, 0.0]")
    print(f"  Total Duration: {target_traj.getTotalDuration():.3f} s")
    print(f"  Position at t=0: {target_traj.getPosition(0.0)}")
    print(f"  Position at t={estimated_duration:.1f}: {target_traj.getPosition(estimated_duration)}")
    
    # ---- Create and Configure Optimizer ----
    optimizer = catching.SimpleCatching()
    
    # Set dynamic limits
    optimizer.setDynamicLimits(
        max_velocity=10.0,
        max_acceleration=10.0,
        max_thrust=20.0,
        min_thrust=2.0,
        max_body_rate=3.0,
        max_yaw_body_rate=2.0
    )
    
    # Set optimization weights
    optimizer.setOptimizationWeights(
        time_weight=5.0,  # Reduced to allow longer trajectories
        position_weight=100.0,
        velocity_weight=10.0,
        acceleration_weight=1.0,
        thrust_weight=1.0,
        body_rate_weight=1.0,
        terminal_velocity_weight=-1.0,
        collision_weight=1.0
    )
    
    # Set trajectory parameters
    optimizer.setTrajectoryParams(
        integration_steps=20,
        traj_pieces_num=3,  # Use 3 pieces for simpler initial test
        time_var_dim=1,
        custom_var_num=0
    )
    
    # Set target trajectory
    optimizer.setTargetTrajectory(target_traj)
    
    # Optionally set catching attitude (not used in optimization yet)
    # euler_angles = np.array([0.0, 0.0, 0.0])
    # optimizer.setCatchingAttitude(euler_angles)
    
    print(f"\nOptimizer configured successfully!")
    
    # ---- Generate Trajectory ----
    print(f"\nGenerating interception trajectory...")
    
    success, trajectory = optimizer.generateTrajectory(initial_state)
    
    if success:
        print(f"✓ Trajectory generation successful!")
        print(f"  Number of pieces: {trajectory.getPieceNum()}")
        print(f"  Total duration: {trajectory.getTotalDuration():.3f} seconds")
        print(f"  Optimization iterations: {optimizer.getIterationCount()}")
        
        # Sample trajectory at various time points
        print(f"\nTrajectory samples:")
        total_duration = trajectory.getTotalDuration()
        for i in range(5):
            t = i * total_duration / 4
            pos = trajectory.getPos(t)
            vel = trajectory.getVel(t)
            acc = trajectory.getAcc(t)
            
            target_pos = target_traj.getPosition(t)
            
            print(f"  t={t:.3f}s:")
            print(f"    Drone pos: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")
            print(f"    Target pos: [{target_pos[0]:6.3f}, {target_pos[1]:6.3f}, {target_pos[2]:6.3f}]")
            print(f"    Distance: {np.linalg.norm(pos - target_pos):.3f} m")
        
        # Check final interception
        final_pos = trajectory.getPos(total_duration)
        final_target_pos = target_traj.getPosition(total_duration)
        final_distance = np.linalg.norm(final_pos - final_target_pos)
        
        print(f"\nFinal interception:")
        print(f"  Drone position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
        print(f"  Target position: [{final_target_pos[0]:.3f}, {final_target_pos[1]:.3f}, {final_target_pos[2]:.3f}]")
        print(f"  Interception error: {final_distance:.6f} m")
        
        # Check trajectory constraints
        print(f"\nTrajectory statistics:")
        print(f"  Max velocity: {trajectory.getMaxVelRate():.3f} m/s")
        print(f"  Max acceleration: {trajectory.getMaxAccRate():.3f} m/s²")
        print(f"  Max thrust: {trajectory.getMaxThrust():.3f} m/s²")
        
    else:
        print(f"✗ Trajectory generation failed!")
        return 1
    
    print(f"\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
