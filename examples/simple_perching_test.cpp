#include "simple_perching.h"
#include <iostream>
#include <Eigen/Dense>

int main() {
    std::cout << "Testing SimplePerching class..." << std::endl;
    
    // Create SimplePerching instance
    SimplePerching perching_optimizer;
    
    // Set dynamic limits
    perching_optimizer.setDynamicLimits(30.0, 30.0, 20.0, 1.0, 3.0, 3.0); // max_vel, max_acc, max_thrust, min_thrust, max_body_rate, max_yaw_body_rate
    
    // Set robot parameters  
    perching_optimizer.setRobotParameters(1.0, 0.2, 0.15, 0.5); // landing_speed_offset, tail_length, body_radius, platform_radius
    
    // Set optimization weights (all 8 parameters)
    perching_optimizer.setOptimizationWeights(1.0, -1.0, 100.0, 10.0, 1.0, 10.0, 1.0, 1.0); // time_w, tail_vel_w, pos_w, vel_w, acc_w, thrust_w, body_rate_w, perching_collision_w
    
    // Define initial state as 3x4 matrix (position, velocity, acceleration, jerk)
    Eigen::MatrixXd initial_state(3, 4);
    initial_state.col(0) = Eigen::Vector3d(0.0, 0.0, 2.0); // position
    initial_state.col(1) = Eigen::Vector3d(2.0, 0.0, 0.0); // velocity
    initial_state.col(2) = Eigen::Vector3d(0.0, 0.0, 0.0); // acceleration
    initial_state.col(3) = Eigen::Vector3d(0.0, 0.0, 0.0); // jerk
    
    // Define target position and velocity
    Eigen::Vector3d target_pos(5.0, 0.0, 1.0);
    Eigen::Vector3d target_vel(1.0, 0.0, 0.0);
    
    // Define landing quaternion (example: slight rotation)
    Eigen::Quaterniond landing_quat(0.9659, 0.0, 0.0, 0.2588); // ~30 degree rotation about z
    
    std::cout << "Initial position: " << initial_state.col(0).transpose() << std::endl;
    std::cout << "Target position: " << target_pos.transpose() << std::endl;
    std::cout << "Target velocity: " << target_vel.transpose() << std::endl;
    
    try {
        // Generate trajectory
        Trajectory trajectory;
        std::cout << "About to generate trajectory..." << std::endl;
        bool success = perching_optimizer.generateTrajectory(
            initial_state,
            target_pos, target_vel,
            landing_quat,
            8, // num_pieces
            trajectory
        );
        std::cout << "Generation completed with result: " << (success ? "success" : "failure") << std::endl;
        
        if (success) {
            std::cout << "✓ Trajectory generation successful!" << std::endl;
            
            // Get trajectory duration
            double total_time = perching_optimizer.getOptimizedTotalDuration();
            std::cout << "Total trajectory time: " << total_time << " seconds" << std::endl;
            
            // Sample some points along the trajectory
            std::cout << "\nSample trajectory points:" << std::endl;
            for (int i = 0; i <= 4; ++i) {
                double t = (total_time * i) / 4.0;
                Eigen::Vector3d pos = trajectory.getPos(t);
                Eigen::Vector3d vel = trajectory.getVel(t);
                Eigen::Vector3d acc = trajectory.getAcc(t);
                std::cout << "t=" << t << "s: pos=[" << pos.transpose() 
                         << "] vel=[" << vel.transpose() << "]" << std::endl;
            }
            
        } else {
            std::cout << "✗ Trajectory generation failed" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "✗ Exception during trajectory generation: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nSimplePerching test completed successfully!" << std::endl;
    return 0;
}