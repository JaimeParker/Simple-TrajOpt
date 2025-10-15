#!/usr/bin/env python3
"""
Catching Optimizer Trajectory Visualization Script

Visualizes the catching optimizer trajectory showing the drone approaching
a stationary target position.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

def load_trajectory_data(filename):
    """Load trajectory data from CSV file."""
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        trajectory = {
            'time': data[:, 0],
            'pos_x': data[:, 1],
            'pos_y': data[:, 2],
            'pos_z': data[:, 3],
            'vel_x': data[:, 4],
            'vel_y': data[:, 5],
            'vel_z': data[:, 6],
            'acc_x': data[:, 7],
            'acc_y': data[:, 8],
            'acc_z': data[:, 9]
        }
        return trajectory
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return None


def create_catching_visualization(traj_data):
    """Create comprehensive visualization for catching trajectory."""
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Color scheme
    color_traj = '#1f77b4'  # Blue for trajectory
    color_target = '#d62728'  # Red for target
    
    # Target position (stationary)
    target_pos = np.array([5.0, 5.0, 1.5])
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Plot trajectory
    ax1.plot(traj_data['pos_x'], traj_data['pos_y'], traj_data['pos_z'], 
             color=color_traj, linewidth=2, label='Catching Trajectory', alpha=0.8)
    
    # Mark start and end points
    ax1.scatter(traj_data['pos_x'][0], traj_data['pos_y'][0], traj_data['pos_z'][0], 
                color='green', s=150, marker='o', label='Start', zorder=10, edgecolors='black', linewidths=2)
    ax1.scatter(traj_data['pos_x'][-1], traj_data['pos_y'][-1], traj_data['pos_z'][-1], 
                color='blue', s=150, marker='s', label='Final Position', zorder=10, edgecolors='black', linewidths=2)
    
    # Mark target position
    ax1.scatter(*target_pos, color=color_target, s=200, marker='*', 
                label='Target Position', zorder=10, edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('X Position (m)', fontsize=10)
    ax1.set_ylabel('Y Position (m)', fontsize=10)
    ax1.set_zlabel('Z Position (m)', fontsize=10)
    ax1.set_title('3D Catching Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Position components over time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(traj_data['time'], traj_data['pos_x'], color=color_traj, linewidth=2, label='X')
    ax2.plot(traj_data['time'], traj_data['pos_y'], color=color_traj, linewidth=2, 
             linestyle='--', label='Y')
    ax2.plot(traj_data['time'], traj_data['pos_z'], color=color_traj, linewidth=2, 
             linestyle=':', label='Z')
    
    # Add target lines
    ax2.axhline(y=target_pos[0], color=color_target, linestyle='-', alpha=0.5, label='Target X')
    ax2.axhline(y=target_pos[1], color=color_target, linestyle='--', alpha=0.5, label='Target Y')
    ax2.axhline(y=target_pos[2], color=color_target, linestyle=':', alpha=0.5, label='Target Z')
    
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Position (m)', fontsize=10)
    ax2.set_title('Position Components vs Time', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Velocity magnitude over time
    ax3 = fig.add_subplot(2, 3, 3)
    vel_mag = np.sqrt(traj_data['vel_x']**2 + traj_data['vel_y']**2 + traj_data['vel_z']**2)
    ax3.plot(traj_data['time'], vel_mag, color=color_traj, linewidth=2)
    ax3.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='Limit (10 m/s)')
    
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Velocity Magnitude (m/s)', fontsize=10)
    ax3.set_title('Velocity Magnitude vs Time', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Acceleration magnitude over time
    ax4 = fig.add_subplot(2, 3, 4)
    acc_mag = np.sqrt(traj_data['acc_x']**2 + traj_data['acc_y']**2 + traj_data['acc_z']**2)
    ax4.plot(traj_data['time'], acc_mag, color=color_traj, linewidth=2)
    ax4.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='Limit (10 m/s²)')
    
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Acceleration Magnitude (m/s²)', fontsize=10)
    ax4.set_title('Acceleration Magnitude vs Time', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Distance to target over time
    ax5 = fig.add_subplot(2, 3, 5)
    dist_to_target = np.sqrt((traj_data['pos_x'] - target_pos[0])**2 + 
                             (traj_data['pos_y'] - target_pos[1])**2 + 
                             (traj_data['pos_z'] - target_pos[2])**2)
    ax5.plot(traj_data['time'], dist_to_target, color='purple', linewidth=2)
    ax5.fill_between(traj_data['time'], dist_to_target, alpha=0.3, color='purple')
    
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('Distance to Target (m)', fontsize=10)
    ax5.set_title('Distance to Target vs Time', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add statistics text
    final_dist = dist_to_target[-1]
    ax5.text(0.02, 0.98, f'Initial: {dist_to_target[0]:.3f}m\nFinal: {final_dist:.6f}m', 
             transform=ax5.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Top view (X-Y plane)
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Plot trajectory in X-Y plane
    ax6.plot(traj_data['pos_x'], traj_data['pos_y'], color=color_traj, linewidth=2, 
             label='Trajectory', alpha=0.8)
    
    # Mark key points
    ax6.scatter(traj_data['pos_x'][0], traj_data['pos_y'][0], 
                color='green', s=150, marker='o', label='Start', zorder=10, 
                edgecolors='black', linewidths=2)
    ax6.scatter(traj_data['pos_x'][-1], traj_data['pos_y'][-1], 
                color='blue', s=150, marker='s', label='Final', zorder=10,
                edgecolors='black', linewidths=2)
    ax6.scatter(target_pos[0], target_pos[1], 
                color=color_target, s=200, marker='*', label='Target', zorder=10,
                edgecolors='black', linewidths=2)
    
    ax6.set_xlabel('X Position (m)', fontsize=10)
    ax6.set_ylabel('Y Position (m)', fontsize=10)
    ax6.set_title('Top View (X-Y Plane)', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


def print_trajectory_statistics(traj_data):
    """Print detailed statistics about the trajectory."""
    print("\n" + "="*60)
    print("CATCHING TRAJECTORY STATISTICS")
    print("="*60)
    
    print(f"\n📊 Basic Statistics:")
    print(f"  Duration: {traj_data['time'][-1]:.3f}s")
    print(f"  Data points: {len(traj_data['time'])}")
    print(f"  Sample rate: {1.0 / (traj_data['time'][1] - traj_data['time'][0]):.1f} Hz")
    
    print(f"\n📍 Position:")
    print(f"  Start: ({traj_data['pos_x'][0]:.3f}, {traj_data['pos_y'][0]:.3f}, {traj_data['pos_z'][0]:.3f})")
    print(f"  End:   ({traj_data['pos_x'][-1]:.3f}, {traj_data['pos_y'][-1]:.3f}, {traj_data['pos_z'][-1]:.3f})")
    
    target_pos = np.array([5.0, 5.0, 1.5])
    final_error = np.sqrt((traj_data['pos_x'][-1] - target_pos[0])**2 + 
                         (traj_data['pos_y'][-1] - target_pos[1])**2 + 
                         (traj_data['pos_z'][-1] - target_pos[2])**2)
    print(f"  Target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
    print(f"  Final error: {final_error:.6f}m")
    
    # Calculate velocity and acceleration magnitudes
    vel_mag = np.sqrt(traj_data['vel_x']**2 + traj_data['vel_y']**2 + traj_data['vel_z']**2)
    acc_mag = np.sqrt(traj_data['acc_x']**2 + traj_data['acc_y']**2 + traj_data['acc_z']**2)
    
    print(f"\n🚀 Dynamic Characteristics:")
    print(f"  Max velocity: {vel_mag.max():.3f} m/s")
    print(f"  Max acceleration: {acc_mag.max():.3f} m/s²")
    print(f"  Mean velocity: {vel_mag.mean():.3f} m/s")
    print(f"  Mean acceleration: {acc_mag.mean():.3f} m/s²")
    
    print(f"\n⚖️  Constraint Compliance:")
    vel_limit = 10.0
    acc_limit = 10.0
    print(f"  Velocity limit ({vel_limit} m/s): {'✅ OK' if vel_mag.max() <= vel_limit else '❌ VIOLATED'}")
    print(f"  Acceleration limit ({acc_limit} m/s²): {'✅ OK' if acc_mag.max() <= acc_limit else '❌ VIOLATED'}")
    
    print(f"\n📏 Path Length:")
    path_segments = np.sqrt(np.diff(traj_data['pos_x'])**2 + 
                           np.diff(traj_data['pos_y'])**2 + 
                           np.diff(traj_data['pos_z'])**2)
    total_path_length = np.sum(path_segments)
    straight_line_dist = np.sqrt((traj_data['pos_x'][-1] - traj_data['pos_x'][0])**2 + 
                                 (traj_data['pos_y'][-1] - traj_data['pos_y'][0])**2 + 
                                 (traj_data['pos_z'][-1] - traj_data['pos_z'][0])**2)
    print(f"  Total path length: {total_path_length:.3f}m")
    print(f"  Straight line distance: {straight_line_dist:.3f}m")
    print(f"  Path efficiency: {(straight_line_dist / total_path_length * 100):.1f}%")


def main():
    """Main function to visualize catching trajectory."""
    print("🚁 Catching Optimizer Trajectory Visualization")
    print("=" * 50)
    
    # Load trajectory data
    traj_data = load_trajectory_data('../assets/catching_optimizer_trajectory.csv')
    
    if traj_data is None:
        print("❌ Failed to load trajectory data")
        print("Make sure to run test_catching_optimizer.py first!")
        return 1
    
    print("✅ Trajectory data loaded successfully")
    
    # Print statistics
    print_trajectory_statistics(traj_data)
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    fig = create_catching_visualization(traj_data)
    
    # Save figure
    output_file = '../assets/catching_trajectory_visualization.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_file}")
    
    # Show interactive plot
    print("\n📊 Displaying interactive plot...")
    plt.show()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
