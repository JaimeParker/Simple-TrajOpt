#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "catching_optimizer.h"
#include "perching_optimizer.h"
#include "simple_trajectory.h"

namespace {

bool compareTrajectories(const Trajectory& traj1, const Trajectory& traj2, double tolerance = 1e-6) {
    if (std::abs(traj1.getTotalDuration() - traj2.getTotalDuration()) > tolerance) {
        std::cout << "Duration mismatch: " << traj1.getTotalDuration() << " vs "
                  << traj2.getTotalDuration() << std::endl;
        return false;
    }

    if (traj1.getPieceNum() != traj2.getPieceNum()) {
        std::cout << "Piece count mismatch: " << traj1.getPieceNum() << " vs "
                  << traj2.getPieceNum() << std::endl;
        return false;
    }

    const double duration = traj1.getTotalDuration();
    const int num_samples = 20;

    for (int i = 0; i <= num_samples; ++i) {
        double t = (static_cast<double>(i) / num_samples) * duration;

        Eigen::Vector3d pos1 = traj1.getPos(t);
        Eigen::Vector3d pos2 = traj2.getPos(t);
        Eigen::Vector3d vel1 = traj1.getVel(t);
        Eigen::Vector3d vel2 = traj2.getVel(t);
        Eigen::Vector3d acc1 = traj1.getAcc(t);
        Eigen::Vector3d acc2 = traj2.getAcc(t);

        if ((pos1 - pos2).norm() > tolerance || (vel1 - vel2).norm() > tolerance ||
            (acc1 - acc2).norm() > tolerance) {
            std::cout << "Trajectory mismatch at t=" << t << std::endl;
            std::cout << "  Position diff: " << (pos1 - pos2).transpose() << std::endl;
            std::cout << "  Velocity diff: " << (vel1 - vel2).transpose() << std::endl;
            std::cout << "  Acceleration diff: " << (acc1 - acc2).transpose() << std::endl;
            return false;
        }
    }

    return true;
}

void printTrajectoryInfo(const std::string& name, const Trajectory& traj) {
    std::cout << name << " Trajectory Info:" << std::endl;
    std::cout << "  Duration: " << traj.getTotalDuration() << " seconds" << std::endl;
    std::cout << "  Pieces: " << traj.getPieceNum() << std::endl;

    Eigen::Vector3d start_pos = traj.getPos(0.0);
    Eigen::Vector3d start_vel = traj.getVel(0.0);
    Eigen::Vector3d end_pos = traj.getPos(traj.getTotalDuration());
    Eigen::Vector3d end_vel = traj.getVel(traj.getTotalDuration());

    std::cout << "  Start: pos=" << start_pos.transpose() << ", vel=" << start_vel.transpose()
              << std::endl;
    std::cout << "  End:   pos=" << end_pos.transpose() << ", vel=" << end_vel.transpose()
              << std::endl;
}

}  // namespace

int main() {
    std::cout << "=== Catching vs Perching Optimizer Comparison ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    // Common initial and terminal states
    DroneState initial_state;
    initial_state.position = Eigen::Vector3d(0.0, 0.0, 1.0);
    initial_state.velocity = Eigen::Vector3d(0.0, 0.0, 0.0);
    initial_state.acceleration = Eigen::Vector3d(0.0, 0.0, 0.0);
    initial_state.jerk = Eigen::Vector3d::Zero();
    initial_state.attitude_quat = Eigen::Quaterniond::Identity();

    DroneState terminal_state;
    terminal_state.position = Eigen::Vector3d(5.0, 5.0, 1.5);
    terminal_state.velocity = Eigen::Vector3d::Zero();
    terminal_state.acceleration = Eigen::Vector3d::Zero();
    terminal_state.jerk = Eigen::Vector3d::Zero();
    terminal_state.attitude_quat = Eigen::Quaterniond::Identity();

    int num_pieces = 3;
    Eigen::Quaterniond landing_quat = Eigen::Quaterniond::Identity();

    // Catching optimizer setup
    CatchingOptimizer catching_optimizer;
    catching_optimizer.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
    catching_optimizer.setOptimizationWeights(1.0, 100.0, 10.0, 1.0, 1.0, 1.0, -1.0, 1.0);
    catching_optimizer.setTrajectoryParams(20, num_pieces, 1, 3);
    catching_optimizer.setInitialState(initial_state);
    catching_optimizer.setTerminalState(terminal_state);

    auto target_traj = std::make_shared<DiscreteTrajectory>();
    target_traj->addWaypoint({0.0, terminal_state.position, terminal_state.velocity, Eigen::Vector3d::Zero()});
    target_traj->addWaypoint({12.0, terminal_state.position, terminal_state.velocity, Eigen::Vector3d::Zero()});
    catching_optimizer.setTargetTrajectory(target_traj);

    Trajectory catching_traj;
    std::cout << "Generating CatchingOptimizer trajectory..." << std::endl;
    bool catching_success = catching_optimizer.generateTrajectory(catching_traj);

    // Perching optimizer setup
    traj_opt::PerchingOptimizer perching_optimizer;
    perching_optimizer.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
    perching_optimizer.setRobotParameters(1.0, 0.3, 0.1, 0.5);
    perching_optimizer.setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    perching_optimizer.setIntegrationSteps(20);

    Eigen::MatrixXd initial_matrix(3, 4);
    initial_matrix.col(0) = initial_state.position;
    initial_matrix.col(1) = initial_state.velocity;
    initial_matrix.col(2) = initial_state.acceleration;
    initial_matrix.col(3) = initial_state.jerk;

    Eigen::Vector3d target_position = terminal_state.position;
    Eigen::Vector3d target_velocity = terminal_state.velocity;

    Trajectory perching_traj;
    std::cout << "Generating PerchingOptimizer trajectory..." << std::endl;
    bool perching_success = perching_optimizer.generateTrajectory(
        initial_matrix, target_position, target_velocity, landing_quat, num_pieces, perching_traj);

    std::cout << "\nResults:" << std::endl;
    std::cout << "CatchingOptimizer success: " << (catching_success ? "YES" : "NO")
              << " (iterations: " << catching_optimizer.getIterationCount() << ")" << std::endl;
    std::cout << "PerchingOptimizer success: " << (perching_success ? "YES" : "NO") << std::endl;

    if (catching_success) {
        printTrajectoryInfo("Catching", catching_traj);
    }
    if (perching_success) {
        printTrajectoryInfo("Perching", perching_traj);
    }

    if (catching_success && perching_success) {
        bool match = compareTrajectories(catching_traj, perching_traj, 1e-1);
        std::cout << "\nTrajectory comparison: " << (match ? "MATCH" : "MISMATCH") << std::endl;
        if (match) {
            std::cout << "✅ Comparison PASSED" << std::endl;
        } else {
            std::cout << "❌ Comparison FAILED" << std::endl;
            std::cout << "Be noticed, after version 292770762807924611dc69d5a558229d019d4674, this comparison become meaningless" << std::endl;
        }
    } else {
        std::cout << "❌ Comparison skipped due to failed optimization" << std::endl;
    }

    return 0;
}
