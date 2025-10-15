#!/usr/bin/env python3
"""
Catching Trajectory Summary Generator

Creates a text-based summary of the catching optimizer trajectory results.
"""

import numpy as np

def load_catching_trajectory():
    """Load catching trajectory and generate summary."""
    
    print("🚁 Catching Trajectory Analysis Summary")
    print("=" * 50)
    
    try:
        # Load catching trajectory
        data = np.loadtxt('../assets/catching_optimizer_trajectory.csv', delimiter=',', skiprows=1)
        traj = {
            'time': data[:, 0],
            'pos': data[:, 1:4],
            'vel': data[:, 4:7], 
            'acc': data[:, 7:10]
        }
        
        print(f"✅ Successfully loaded catching trajectory")
        
    except Exception as e:
        print(f"❌ Error loading trajectory: {e}")
        return
    
    # Target position (stationary)
    target_pos = np.array([5.0, 5.0, 1.5])
    
    # Basic information
    print(f"\n📊 TRAJECTORY INFORMATION:")
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    
    duration = traj['time'][-1]
    print(f"{'Duration (s)':<25} {duration:<15.3f}")
    
    points = len(traj['time'])
    print(f"{'Data Points':<25} {points:<15d}")
    
    sample_rate = 1.0 / (traj['time'][1] - traj['time'][0])
    print(f"{'Sample Rate (Hz)':<25} {sample_rate:<15.1f}")
    
    # Initial and final positions
    initial_pos = traj['pos'][0]
    final_pos = traj['pos'][-1]
    
    print(f"\n🎯 POSITION ANALYSIS:")
    print(f"  Initial Position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
    print(f"  Final Position:   [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    print(f"  Target Position:  [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
    # Error analysis
    final_error = np.linalg.norm(final_pos - target_pos)
    print(f"\n📏 ERROR ANALYSIS:")
    print(f"  Final Position Error: {final_error:.6f}m")
    
    if final_error < 0.001:
        error_assessment = "🎯 EXCELLENT - Reached target with sub-millimeter precision"
    elif final_error < 0.01:
        error_assessment = "✅ VERY GOOD - Reached target with centimeter-level precision"
    elif final_error < 0.1:
        error_assessment = "✅ GOOD - Reached target with decimeter-level precision"
    else:
        error_assessment = "⚠️  MODERATE - Final position deviates from target"
    
    print(f"  Assessment: {error_assessment}")
    
    # Initial distance
    initial_dist = np.linalg.norm(initial_pos - target_pos)
    print(f"  Initial Distance: {initial_dist:.3f}m")
    print(f"  Distance Reduction: {(initial_dist - final_error):.3f}m ({(1 - final_error/initial_dist)*100:.2f}%)")
    
    # Dynamic characteristics
    vel_mag = np.linalg.norm(traj['vel'], axis=1)
    acc_mag = np.linalg.norm(traj['acc'], axis=1)
    
    vel_max = vel_mag.max()
    acc_max = acc_mag.max()
    
    print(f"\n🚀 DYNAMIC CHARACTERISTICS:")
    print(f"  Velocity:")
    print(f"    Maximum:  {vel_max:.3f}m/s")
    print(f"    Mean:     {vel_mag.mean():.3f}m/s")
    print(f"    Final:    {vel_mag[-1]:.6f}m/s")
    print(f"  Acceleration:")
    print(f"    Maximum:  {acc_max:.3f}m/s²")
    print(f"    Mean:     {acc_mag.mean():.3f}m/s²")
    print(f"    Final:    {acc_mag[-1]:.6f}m/s²")
    
    # Constraint compliance
    vel_limit = 10.0
    acc_limit = 10.0
    
    print(f"\n⚖️  CONSTRAINT COMPLIANCE:")
    print(f"  Velocity Limit ({vel_limit}m/s):")
    vel_ok = vel_max <= vel_limit
    print(f"    Status: {'✅ OK' if vel_ok else '❌ VIOLATED'}")
    print(f"    Usage: {(vel_max/vel_limit*100):.1f}%")
    
    print(f"  Acceleration Limit ({acc_limit}m/s²):")
    acc_ok = acc_max <= acc_limit
    print(f"    Status: {'✅ OK' if acc_ok else '❌ VIOLATED'}")
    print(f"    Usage: {(acc_max/acc_limit*100):.1f}%")
    
    # Path analysis
    path_segments = np.linalg.norm(np.diff(traj['pos'], axis=0), axis=1)
    total_path_length = np.sum(path_segments)
    straight_line_dist = np.linalg.norm(final_pos - initial_pos)
    path_efficiency = straight_line_dist / total_path_length
    
    print(f"\n📐 PATH ANALYSIS:")
    print(f"  Total Path Length:      {total_path_length:.3f}m")
    print(f"  Straight Line Distance: {straight_line_dist:.3f}m")
    print(f"  Path Efficiency:        {path_efficiency*100:.1f}%")
    
    if path_efficiency > 0.95:
        path_assessment = "🎯 EXCELLENT - Nearly straight path"
    elif path_efficiency > 0.85:
        path_assessment = "✅ GOOD - Efficient path"
    elif path_efficiency > 0.70:
        path_assessment = "⚠️  MODERATE - Some deviation from straight path"
    else:
        path_assessment = "❌ POOR - Inefficient path"
    
    print(f"  Assessment: {path_assessment}")
    
    # Smoothness analysis
    acc_changes = np.abs(np.diff(acc_mag))
    jerk_approx = acc_changes / (traj['time'][1] - traj['time'][0])
    
    print(f"\n🌊 SMOOTHNESS ANALYSIS:")
    print(f"  Max Jerk (approx): {jerk_approx.max():.3f}m/s³")
    print(f"  Mean Jerk (approx): {jerk_approx.mean():.3f}m/s³")
    
    if jerk_approx.max() < 10.0:
        smoothness = "🎯 EXCELLENT - Very smooth trajectory"
    elif jerk_approx.max() < 50.0:
        smoothness = "✅ GOOD - Smooth trajectory"
    elif jerk_approx.max() < 100.0:
        smoothness = "⚠️  MODERATE - Some abrupt changes"
    else:
        smoothness = "❌ POOR - Trajectory has jerky motions"
    
    print(f"  Assessment: {smoothness}")
    
    # Final velocity analysis
    final_vel = traj['vel'][-1]
    final_vel_mag = np.linalg.norm(final_vel)
    
    print(f"\n🎯 TERMINAL STATE:")
    print(f"  Final Velocity Vector: [{final_vel[0]:.6f}, {final_vel[1]:.6f}, {final_vel[2]:.6f}]")
    print(f"  Final Velocity Magnitude: {final_vel_mag:.6f}m/s")
    
    if final_vel_mag < 0.01:
        terminal_vel_status = "✅ EXCELLENT - Nearly zero final velocity"
    elif final_vel_mag < 0.1:
        terminal_vel_status = "✅ GOOD - Low final velocity"
    else:
        terminal_vel_status = "⚠️  MODERATE - Non-zero final velocity"
    
    print(f"  Status: {terminal_vel_status}")
    
    # Overall assessment
    print(f"\n🏆 OVERALL ASSESSMENT:")
    
    all_ok = vel_ok and acc_ok and final_error < 0.1 and path_efficiency > 0.7
    
    if all_ok and final_error < 0.01:
        overall = "🎉 EXCELLENT: Trajectory meets all requirements with high precision"
    elif all_ok:
        overall = "✅ GOOD: Trajectory meets all requirements"
    elif vel_ok and acc_ok:
        overall = "⚠️  ACCEPTABLE: Constraints satisfied but target accuracy could be improved"
    else:
        overall = "❌ NEEDS IMPROVEMENT: Some constraints violated or poor performance"
    
    print(f"  {overall}")
    
    # Comparison with straight-line motion
    print(f"\n📊 COMPARISON WITH STRAIGHT-LINE MOTION:")
    avg_velocity = straight_line_dist / duration
    time_at_max_vel = straight_line_dist / vel_limit if vel_limit > 0 else float('inf')
    
    print(f"  Average Velocity:       {avg_velocity:.3f}m/s")
    print(f"  Time if at max velocity: {time_at_max_vel:.3f}s")
    print(f"  Time efficiency:        {(time_at_max_vel/duration*100):.1f}%")

if __name__ == "__main__":
    load_catching_trajectory()
