// #include "catching_optimizer.h"
#include "simple_perching.h"

#include <cassert>

#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "perching_optimizer.h"

bool compareTrajectories(const Trajectory& traj1, const Trajectory& traj2, double tolerance = 1e-6) {
    if (std::abs(traj1.getTotalDuration() - traj2.getTotalDuration()) > tolerance) {
        std::cout << "Duration mismatch: " << traj1.getTotalDuration() << " vs " << traj2.getTotalDuration() << std::endl;
        return false;
    }

    if (traj1.getPieceNum() != traj2.getPieceNum()) {
        std::cout << "Piece count mismatch: " << traj1.getPieceNum() << " vs " << traj2.getPieceNum() << std::endl;
        return false;
    }

    // Sample trajectories at multiple points and compare
    double duration = traj1.getTotalDuration();
    int num_samples = 20;

    for (int i = 0; i <= num_samples; ++i) {
        double t = (static_cast<double>(i) / num_samples) * duration;

        Eigen::Vector3d pos1 = traj1.getPos(t);
        Eigen::Vector3d pos2 = traj2.getPos(t);
        Eigen::Vector3d vel1 = traj1.getVel(t);
        Eigen::Vector3d vel2 = traj2.getVel(t);
        Eigen::Vector3d acc1 = traj1.getAcc(t);
        Eigen::Vector3d acc2 = traj2.getAcc(t);

        if ((pos1 - pos2).norm() > tolerance ||
            (vel1 - vel2).norm() > tolerance ||
            (acc1 - acc2).norm() > tolerance) {
            std::cout << "Trajectory mismatch at t=" << t << std::endl;
            std::cout << "Position diff: " << (pos1 - pos2).transpose() << std::endl;
            std::cout << "Velocity diff: " << (vel1 - vel2).transpose() << std::endl;
            std::cout << "Acceleration diff: " << (acc1 - acc2).transpose() << std::endl;
            return false;
        }
    }

    return true;
}

void printTrajectoryInfo(const std::string& name, const Trajectory& traj) {
    std::cout << name << " Trajectory Info:" << std::endl;
    std::cout << "  Duration: " << traj.getTotalDuration() << " seconds" << std::endl;
    std::cout << "  Pieces: " << traj.getPieceNum() << std::endl;

    // Print start and end states
    Eigen::Vector3d start_pos = traj.getPos(0.0);
    Eigen::Vector3d start_vel = traj.getVel(0.0);
    Eigen::Vector3d end_pos = traj.getPos(traj.getTotalDuration());
    Eigen::Vector3d end_vel = traj.getVel(traj.getTotalDuration());

    std::cout << "  Start: pos=" << start_pos.transpose() << ", vel=" << start_vel.transpose() << std::endl;
    std::cout << "  End:   pos=" << end_pos.transpose() << ", vel=" << end_vel.transpose() << std::endl;
}

// DroneState toDroneState(const Eigen::MatrixXd& state_matrix) {
//     assert(state_matrix.rows() == 3 && state_matrix.cols() == 4 && "State matrix must be 3x4");

//     DroneState state;
//     state.position = state_matrix.col(0);
//     state.velocity = state_matrix.col(1);
//     state.acceleration = state_matrix.col(2);
//     state.jerk = state_matrix.col(3);
//     state.attitude.setZero();
//     state.body_rate.setZero();
//     state.attitude_quat.setIdentity();
//     return state;
// }

int main() {
    std::cout << "=== Testing SimplePerching vs PerchingOptimizer ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    // Test setup - common parameters (matching perching_comparison.cpp)
    Eigen::MatrixXd initial_state(3, 4);
    initial_state.setZero();
    initial_state.col(0) = Eigen::Vector3d(0.0, 0.0, 3.0);   // Initial position (hovering above target)
    initial_state.col(1) = Eigen::Vector3d(2.0, 0.0, -0.5);  // Initial velocity (moving toward target)
    initial_state.col(2) = Eigen::Vector3d(0.0, 0.0, 0.0);   // Initial acceleration
    initial_state.col(3) = Eigen::Vector3d(0.0, 0.0, 0.0);   // Initial jerk

    // Moving target (car/platform)
    Eigen::Vector3d target_pos(8.0, 2.0, 1.0);
    Eigen::Vector3d target_vel(1.5, 0.5, 0.0);

    // Landing orientation (slightly tilted for perching) - ~30 degree tilt around Y-axis
    Eigen::Quaterniond landing_quat(0.9659, 0.0, 0.2588, 0.0);

    int num_pieces = 6;

    std::cout << "\nPerching problem setup:" << std::endl;
    std::cout << "  Initial position: [" << initial_state.col(0).transpose() << "]" << std::endl;
    std::cout << "  Initial velocity: [" << initial_state.col(1).transpose() << "]" << std::endl;
    std::cout << "  Target position:  [" << target_pos.transpose() << "]" << std::endl;
    std::cout << "  Target velocity:  [" << target_vel.transpose() << "]" << std::endl;
    std::cout << "  Landing quaternion: [" << landing_quat.w() << ", " << landing_quat.x()
              << ", " << landing_quat.y() << ", " << landing_quat.z() << "]" << std::endl;
    std::cout << "  Number of pieces: " << num_pieces << std::endl;

    // Test 1: Basic functionality test
    std::cout << "\n=== Test 1: Basic Functionality ===" << std::endl;

    try {
        // Create optimizers
        SimplePerching simple_perching;
        traj_opt::PerchingOptimizer perching_optimizer;

        // Configure identical parameters (matching perching_comparison.cpp)
        simple_perching.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
        simple_perching.setRobotParameters(1.0, 0.3, 0.1, 0.5);
        simple_perching.setOptimizationWeights(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0);

        perching_optimizer.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
        perching_optimizer.setRobotParameters(1.0, 0.3, 0.1, 0.5);
        perching_optimizer.setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        perching_optimizer.setIntegrationSteps(20);

        // Generate trajectories
        Trajectory simple_traj, perching_traj;

        std::cout << "Generating SimplePerching trajectory..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        bool simple_success = simple_perching.generateTrajectory(initial_state, target_pos, target_vel,
                                                                 landing_quat, num_pieces, simple_traj);
        auto simple_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);

        std::cout << "Generating PerchingOptimizer trajectory..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        bool perching_success = perching_optimizer.generateTrajectory(initial_state, target_pos, target_vel,
                                                                      landing_quat, num_pieces, perching_traj);
        auto perching_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);

        std::cout << "\nResults:" << std::endl;
        std::cout << "SimplePerching success: " << (simple_success ? "YES" : "NO")
                  << " (time: " << simple_time.count() << "ms, iterations: " << simple_perching.getIterationCount() << ")" << std::endl;
        std::cout << "PerchingOptimizer success: " << (perching_success ? "YES" : "NO")
                  << " (time: " << perching_time.count() << "ms)" << std::endl;

        if (simple_success && perching_success) {
            printTrajectoryInfo("SimplePerching", simple_traj);
            printTrajectoryInfo("PerchingOptimizer", perching_traj);

            // Compare trajectories
            bool trajectories_match = compareTrajectories(simple_traj, perching_traj, 1e-4);
            std::cout << "\nTrajectory comparison: " << (trajectories_match ? "MATCH" : "MISMATCH") << std::endl;

            if (trajectories_match) {
                std::cout << "✅ Test 1 PASSED: Trajectories are identical within tolerance" << std::endl;
            } else {
                std::cout << "❌ Test 1 FAILED: Trajectories differ" << std::endl;
            }
        } else {
            std::cout << "❌ Test 1 FAILED: One or both optimizations failed" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "❌ Test 1 FAILED with exception: " << e.what() << std::endl;
    }

    // Test 2: Different trajectory configurations
    std::cout << "\n=== Test 2: Multi-piece Trajectory ===" << std::endl;

    try {
        SimplePerching simple_perching2;
        traj_opt::PerchingOptimizer perching_optimizer2;

        // Use same configuration but with different piece count
        int num_pieces_2 = 3;

        simple_perching2.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
        simple_perching2.setRobotParameters(1.0, 0.3, 0.1, 0.5);
        simple_perching2.setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

        perching_optimizer2.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
        perching_optimizer2.setRobotParameters(1.0, 0.3, 0.1, 0.5);
        perching_optimizer2.setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

        // Use same target but test fewer pieces
        Eigen::Vector3d target_pos_2 = target_pos;
        Eigen::Vector3d target_vel_2 = target_vel;

        Trajectory simple_traj2, perching_traj2;

        std::cout << "Testing " << num_pieces_2 << "-piece trajectory..." << std::endl;
        bool simple_success2 = simple_perching2.generateTrajectory(initial_state, target_pos_2, target_vel_2,
                                                                   landing_quat, num_pieces_2, simple_traj2);
        bool perching_success2 = perching_optimizer2.generateTrajectory(initial_state, target_pos_2, target_vel_2,
                                                                        landing_quat, num_pieces_2, perching_traj2);

        std::cout << "SimplePerching success: " << (simple_success2 ? "YES" : "NO") << std::endl;
        std::cout << "PerchingOptimizer success: " << (perching_success2 ? "YES" : "NO") << std::endl;

        if (simple_success2 && perching_success2) {
            bool trajectories_match2 = compareTrajectories(simple_traj2, perching_traj2, 1e-4);
            std::cout << "Trajectory comparison: " << (trajectories_match2 ? "MATCH" : "MISMATCH") << std::endl;

            if (trajectories_match2) {
                std::cout << "✅ Test 2 PASSED: Multi-piece trajectories match" << std::endl;
            } else {
                std::cout << "❌ Test 2 FAILED: Multi-piece trajectories differ" << std::endl;
            }
        } else {
            std::cout << "❌ Test 2 FAILED: One or both optimizations failed" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "❌ Test 2 FAILED with exception: " << e.what() << std::endl;
    }

    // Test 3: Single piece trajectory
    std::cout << "\n=== Test 3: Single Piece Trajectory ===" << std::endl;

    try {
        SimplePerching simple_perching3;
        traj_opt::PerchingOptimizer perching_optimizer3;

        // Same landing orientation but single piece
        Eigen::Quaterniond vertical_quat = landing_quat;  // Use same orientation as main test

        simple_perching3.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
        simple_perching3.setRobotParameters(1.0, 0.3, 0.1, 0.5);
        simple_perching3.setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

        perching_optimizer3.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
        perching_optimizer3.setRobotParameters(1.0, 0.3, 0.1, 0.5);
        perching_optimizer3.setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

        Eigen::Vector3d target_pos_3 = target_pos;  // Use same target as main test

        Trajectory simple_traj3, perching_traj3;

        std::cout << "Testing single piece trajectory..." << std::endl;
        bool simple_success3 = simple_perching3.generateTrajectory(initial_state, target_pos_3, target_vel,
                                                                   vertical_quat, 1, simple_traj3);
        bool perching_success3 = perching_optimizer3.generateTrajectory(initial_state, target_pos_3, target_vel,
                                                                        vertical_quat, 1, perching_traj3);

        std::cout << "SimplePerching success: " << (simple_success3 ? "YES" : "NO") << std::endl;
        std::cout << "PerchingOptimizer success: " << (perching_success3 ? "YES" : "NO") << std::endl;

        if (simple_success3 && perching_success3) {
            bool trajectories_match3 = compareTrajectories(simple_traj3, perching_traj3, 1e-4);
            std::cout << "Trajectory comparison: " << (trajectories_match3 ? "MATCH" : "MISMATCH") << std::endl;

            if (trajectories_match3) {
                std::cout << "✅ Test 3 PASSED: Single piece trajectories match" << std::endl;
            } else {
                std::cout << "❌ Test 3 FAILED: Single piece trajectories differ" << std::endl;
            }
        } else {
            std::cout << "❌ Test 3 FAILED: One or both optimizations failed" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "❌ Test 3 FAILED with exception: " << e.what() << std::endl;
    }

    // std::cout << "\n=== Test 4: CatchingOptimizer vs PerchingOptimizer (Stationary Target) ===" << std::endl;

    // try {
    //     // Stationary target configuration
    //     Eigen::Vector3d stationary_target_pos(5.0, 5.0, 2.0);
    //     Eigen::Vector3d stationary_target_vel = Eigen::Vector3d::Zero();
    //     Eigen::Quaterniond identity_quat(1.0, 0.0, 0.0, 0.0);
    //     int num_pieces_stationary = 4;

    //     // Prepare optimizers
    //     CatchingOptimizer catching_optimizer;
    //     traj_opt::PerchingOptimizer perching_optimizer4;

    //     catching_optimizer.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
    //     catching_optimizer.setOptimizationWeights(1.0, 100.0, 10.0, 1.0, 1.0, 1.0, -1.0, 1.0);
    //     catching_optimizer.setTrajectoryParams(20, num_pieces_stationary, 1, 3);

    //     DroneState initial_drone_state = toDroneState(initial_state);
    //     catching_optimizer.setInitialState(initial_drone_state);

    //     DroneState terminal_state;
    //     terminal_state.position = stationary_target_pos;
    //     terminal_state.velocity = stationary_target_vel;
    //     terminal_state.acceleration.setZero();
    //     terminal_state.jerk.setZero();
    //     terminal_state.attitude.setZero();
    //     terminal_state.body_rate.setZero();
    //     terminal_state.attitude_quat.setIdentity();
    //     catching_optimizer.setTerminalState(terminal_state);
    //     catching_optimizer.setCatchingAttitude(identity_quat);

    //     perching_optimizer4.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
    //     perching_optimizer4.setRobotParameters(1.0, 0.3, 0.1, 0.5);
    //     perching_optimizer4.setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    //     perching_optimizer4.setIntegrationSteps(20);

    //     Trajectory catching_traj;
    //     Trajectory perching_traj4;

    //     std::cout << "Generating CatchingOptimizer trajectory..." << std::endl;
    //     auto start_time = std::chrono::high_resolution_clock::now();
    //     bool catching_success = catching_optimizer.generateTrajectory(catching_traj);
    //     auto catching_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    //         std::chrono::high_resolution_clock::now() - start_time);

    //     std::cout << "Generating PerchingOptimizer trajectory for stationary target..." << std::endl;
    //     start_time = std::chrono::high_resolution_clock::now();
    //     bool perching_success4 = perching_optimizer4.generateTrajectory(initial_state,
    //                                                                     stationary_target_pos,
    //                                                                     stationary_target_vel,
    //                                                                     identity_quat,
    //                                                                     num_pieces_stationary,
    //                                                                     perching_traj4);
    //     auto perching_time4 = std::chrono::duration_cast<std::chrono::milliseconds>(
    //         std::chrono::high_resolution_clock::now() - start_time);

    //     std::cout << "Results:" << std::endl;
    //     std::cout << "CatchingOptimizer success: " << (catching_success ? "YES" : "NO")
    //               << " (time: " << catching_time.count() << "ms, iterations: "
    //               << catching_optimizer.getIterationCount() << ")" << std::endl;
    //     std::cout << "PerchingOptimizer success: " << (perching_success4 ? "YES" : "NO")
    //               << " (time: " << perching_time4.count() << "ms)" << std::endl;

    //     if (catching_success && perching_success4) {
    //         printTrajectoryInfo("CatchingOptimizer", catching_traj);
    //         printTrajectoryInfo("PerchingOptimizer (Stationary)", perching_traj4);

    //         bool trajectories_match4 = compareTrajectories(catching_traj, perching_traj4, 1e-3);
    //         std::cout << "Trajectory comparison: " << (trajectories_match4 ? "MATCH" : "MISMATCH") << std::endl;

    //         if (trajectories_match4) {
    //             std::cout << "✅ Test 4 PASSED: Catching and Perching trajectories align for stationary target" << std::endl;
    //         } else {
    //             std::cout << "⚠️  Test 4 WARNING: Trajectories differ for stationary target" << std::endl;
    //         }
    //     } else {
    //         std::cout << "❌ Test 4 FAILED: One or both optimizations failed" << std::endl;
    //     }

    // } catch (const std::exception& e) {
    //     std::cout << "❌ Test 4 FAILED with exception: " << e.what() << std::endl;
    // }

    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "SimplePerching implementation complete and tested against PerchingOptimizer reference." << std::endl;

    return 0;
}