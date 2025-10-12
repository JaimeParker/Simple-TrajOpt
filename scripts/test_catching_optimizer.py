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

#FIXME: since the SimpleTrajectory is an abstract base class, you might need to consider using DiscreteTrajectory, the derived class of it. You need to handle the python interface first, then see how to implement this class.
# After you resolve the issue, you can remove this comment.
# Define a simple target trajectory class
class LinearTargetTrajectory(catching.DiscreteTrajectory):
    """A simple linear trajectory for the target"""
    
    def __init__(self, initial_pos, velocity):
        super().__init__()
        self.initial_pos = np.array(initial_pos)
        self.velocity = np.array(velocity)
    
    def getPosition(self, t):
        """Get target position at time t"""
        return self.initial_pos + self.velocity * t
    
    def getVelocity(self, t):
        """Get target velocity at time t (constant)"""
        return self.velocity
    
    def getAcceleration(self, t):
        """Get target acceleration at time t (zero for linear motion)"""
        return np.array([0.0, 0.0, 0.0])


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
    
    # Target trajectory (moving linearly)
    target_initial_pos = [5.0, 5.0, 1.0]
    target_velocity = [-1.0, -0.5, 0.0]
    
    target_traj = LinearTargetTrajectory(target_initial_pos, target_velocity)
    
    print(f"\nTarget Trajectory:")
    print(f"  Initial Position: {target_initial_pos}")
    print(f"  Velocity: {target_velocity}")
    print(f"  Position at t=0: {target_traj.getPosition(0.0)}")
    print(f"  Position at t=3: {target_traj.getPosition(3.0)}")
    
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
        time_weight=1.0,
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
        traj_pieces_num=3,  # Use 3 pieces for better path quality
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
