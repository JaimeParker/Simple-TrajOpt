//
// Created by Zhaohong Liu on 25-9-30.
//

#ifndef SIMPLE_CATCHING_H
#define SIMPLE_CATCHING_H

#include "simple_trajopt.h"

// A simple parameter struct to pass optimizer variables to computeFinalState
struct CatchingComputeParams : public BaseComputeParams {
    const double* vars;
    double total_duration;
};

// A class to optimize a trajectory to intercept a target with a known flight path.
class SimpleCatching : public SimpleTrajOpt {
   public:
    SimpleCatching() : SimpleTrajOpt() {
        // Initialize with no custom variables for catching (unlike perching which uses tail angle)
        params_.custom_var_dim = 0;
        has_initial_guess_ = false;
    }
    
    ~SimpleCatching() = default;

    // Sets the trajectory of the target to be intercepted.
    void setTargetTrajectory(std::shared_ptr<SimpleTrajectory> target_trajectory) {
        target_traj_ = target_trajectory;
    }

    // Sets the desired final attitude of the drone at the moment of interception.
    void setCatchingAttitude(const Eigen::Vector3d& euler_attitude) {
        catching_att_ = euler2Quaternion(euler_attitude);
    }
    
    void setCatchingAttitude(const Eigen::Quaterniond& quat_attitude) {
        // TODO: Implement logic to use this attitude in the final state cost/constraints.
        // For now, it's stored but not used in the optimization logic below.
        catching_att_ = quat_attitude;
    }

    // Provides an initial guess trajectory to warm-start the optimization.
    void setInitialGuess(std::shared_ptr<Trajectory> pursuer_trajectory) {
        initial_guess_traj_ = pursuer_trajectory;
        has_initial_guess_ = true;
    }

    /**
     * @brief Generates the interception trajectory.
     *
     * @param initial_state The starting state (pos, vel, acc) of the drone.
     * @param trajectory The output optimized trajectory.
     * @return true if the optimization was successful, false otherwise.
     */
    bool generateTrajectory(const DroneState& initial_state, Trajectory& trajectory) override {
        setInitialState(initial_state);
        if (!target_traj_) {
            // Target trajectory must be set before optimization.
            return false;
        }

        // Prepare optimizer variables and L-BFGS parameters.
        preProcessOptUtils();

        // --- Generate Initial Guess ---
        double initial_total_duration = 0.0;
        Eigen::Map<Eigen::MatrixXd> waypoints(optimization_vars_ + params_.time_var_dim, 3, params_.waypoint_num);

        if (has_initial_guess_ && initial_guess_traj_ && initial_guess_traj_->getPieceNum() > 0) {
            // Use the provided initial guess trajectory
            initial_total_duration = initial_guess_traj_->getTotalDuration();
            for (int i = 0; i < params_.waypoint_num; ++i) {
                double ratio = static_cast<double>(i + 1) / static_cast<double>(params_.traj_pieces_num);
                waypoints.col(i) = initial_guess_traj_->getPos(ratio * initial_total_duration);
            }
        } else {
            // Default to a straight-line guess toward the target's initial position
            Eigen::Vector3d target_initial_pos = target_traj_->getPosition(0.0);
            initial_total_duration = (target_initial_pos - initial_state_.position).norm() / params_.max_velocity;
            if (initial_total_duration < 0.5) {
                initial_total_duration = 0.5; // Avoid overly short initial durations
            }
            for (int i = 0; i < params_.waypoint_num; ++i) {
                double ratio = static_cast<double>(i + 1) / static_cast<double>(params_.traj_pieces_num);
                waypoints.col(i) = initial_state_.position + ratio * (target_initial_pos - initial_state_.position);
            }
        }
        optimization_vars_[0] = logC2(initial_total_duration / params_.traj_pieces_num);
        iteration_count_ = 0;

        // --- Run L-BFGS Optimization ---
        double min_objective = 0.0;
        int opt_result = lbfgs::lbfgs_optimize(
            params_.time_var_dim + 3 * params_.waypoint_num + params_.custom_var_dim,
            optimization_vars_,
            &min_objective,
            _objectiveFunction,
            nullptr,
            nullptr, // No early exit callback for now
            this,
            &lbfgs_params_);

        if (opt_result < 0) {
            // Optimization failed.
            return false;
        }

        // --- Finalize Trajectory ---
        double final_piece_duration = expC2(optimization_vars_[0]);
        double final_total_duration = params_.traj_pieces_num * final_piece_duration;
        Eigen::Map<const Eigen::MatrixXd> final_waypoints(optimization_vars_ + params_.time_var_dim, 3, params_.waypoint_num);

        CatchingComputeParams final_params;
        final_params.vars = optimization_vars_;
        final_params.total_duration = final_total_duration;
        DroneState final_state = computeFinalState(final_params);
        
        Eigen::MatrixXd initial_state_matrix(3, 4);
        initial_state_matrix.col(0) = initial_state_.position;
        initial_state_matrix.col(1) = initial_state_.velocity;
        initial_state_matrix.col(2) = initial_state_.acceleration;
        initial_state_matrix.col(3) = initial_state_.jerk;

        Eigen::MatrixXd final_state_matrix(3, 4);
        final_state_matrix.col(0) = final_state.position;
        final_state_matrix.col(1) = final_state.velocity;
        final_state_matrix.col(2) = final_state.acceleration;
        final_state_matrix.col(3) = final_state.jerk;

        minco_optimizer_.generate(initial_state_matrix, final_state_matrix, final_waypoints, final_piece_duration);
        trajectory = minco_optimizer_.getTraj();

        return true;
    }

   protected:
    /**
     * @brief Computes the desired final state by querying the target's trajectory.
     */
    DroneState computeFinalState(const BaseComputeParams& params) override {
        const auto& catching_params = static_cast<const CatchingComputeParams&>(params);
        double total_duration = catching_params.total_duration;

        DroneState final_state;
        final_state.position = target_traj_->getPosition(total_duration);
        final_state.velocity = target_traj_->getVelocity(total_duration);
        final_state.acceleration = target_traj_->getAcceleration(total_duration);
        final_state.jerk.setZero(); // Assume target jerk is zero

        return final_state;
    }

    /**
     * @brief Adds penalties for violating dynamic limits over the trajectory duration.
     */
    void addTimeIntegralPenalty(double& cost) override {
        Eigen::Vector3d pos, vel, acc, jer, snap;
        Eigen::Vector3d grad_temp_pos, grad_temp_vel, grad_temp_acc, grad_temp_jer;
        double cost_temp = 0.0;

        Eigen::Matrix<double, 8, 1> beta0, beta1, beta2, beta3, beta4;
        double sigma1 = 0.0, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7;
        double piece_duration = minco_optimizer_.t(1);
        double step = piece_duration / params_.integration_steps;

        for (int j = 0; j <= params_.integration_steps; ++j) {
            sigma2 = sigma1 * sigma1;
            sigma3 = sigma2 * sigma1;
            sigma4 = sigma2 * sigma2;
            sigma5 = sigma4 * sigma1;
            sigma6 = sigma4 * sigma2;
            sigma7 = sigma4 * sigma3;
            beta0 << 1.0, sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7;
            beta1 << 0.0, 1.0, 2.0 * sigma1, 3.0 * sigma2, 4.0 * sigma3, 5.0 * sigma4, 6.0 * sigma5, 7.0 * sigma6;
            beta2 << 0.0, 0.0, 2.0, 6.0 * sigma1, 12.0 * sigma2, 20.0 * sigma3, 30.0 * sigma4, 42.0 * sigma5;
            beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * sigma1, 60.0 * sigma2, 120.0 * sigma3, 210.0 * sigma4;
            beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * sigma1, 360.0 * sigma2, 840.0 * sigma3;

            double integration_weight = (j == 0 || j == params_.integration_steps) ? 0.5 : 1.0;
            double alpha = static_cast<double>(j) / params_.integration_steps;

            for (int i = 0; i < params_.traj_pieces_num; ++i) {
                const auto& coeff_block = minco_optimizer_.c.block<8, 3>(i * 8, 0);
                pos = coeff_block.transpose() * beta0;
                vel = coeff_block.transpose() * beta1;
                acc = coeff_block.transpose() * beta2;
                jer = coeff_block.transpose() * beta3;
                snap = coeff_block.transpose() * beta4;

                Eigen::Matrix<double, 8, 3> grad_coeff;
                grad_coeff.setZero();
                double grad_time_term = 0.0;
                double cost_inner = 0.0;

                if (computeFloorCost(pos, grad_temp_pos, cost_temp)) {
                    grad_coeff += beta0 * grad_temp_pos.transpose();
                    grad_time_term += grad_temp_pos.dot(vel);
                    cost_inner += cost_temp;
                }
                
                computeVelocityCost(vel, grad_temp_vel, cost_temp);
                if (cost_temp > 0) {
                    grad_coeff += beta1 * grad_temp_vel.transpose();
                    grad_time_term += grad_temp_vel.dot(acc);
                    cost_inner += cost_temp;
                }
                
                if (computeThrustCost(acc, grad_temp_acc, cost_temp)) {
                    grad_coeff += beta2 * grad_temp_acc.transpose();
                    grad_time_term += grad_temp_acc.dot(jer);
                    cost_inner += cost_temp;
                }
                
                if (computeBodyRateCost(acc, jer, grad_temp_acc, grad_temp_jer, cost_temp)) {
                    grad_coeff += beta2 * grad_temp_acc.transpose();
                    grad_coeff += beta3 * grad_temp_jer.transpose();
                    grad_time_term += grad_temp_acc.dot(jer);
                    grad_time_term += grad_temp_jer.dot(snap);
                    cost_inner += cost_temp;
                }

                minco_optimizer_.gdC.block<8, 3>(i * 8, 0) += integration_weight * step * grad_coeff;
                minco_optimizer_.gdT += integration_weight * (cost_inner / params_.integration_steps + alpha * step * grad_time_term);
                cost += integration_weight * step * cost_inner;
            }
            sigma1 += step;
        }
    }

    /**
     * @brief The core objective function that provides cost and gradients to L-BFGS.
     */
    double objectiveFunction(void* ptr, const double* vars, double* grads, int n) override {
        // This is the non-static instance method.
        iteration_count_++;
        const double& log_time_var = vars[0];
        Eigen::Map<const Eigen::MatrixXd> intermediate_waypoints(vars + params_.time_var_dim, 3, params_.waypoint_num);

        double piece_duration = expC2(log_time_var);
        double total_duration = params_.traj_pieces_num * piece_duration;

        CatchingComputeParams current_params;
        current_params.vars = vars;
        current_params.total_duration = total_duration;
        DroneState final_state = computeFinalState(current_params);
        
        Eigen::MatrixXd initial_state_matrix(3, 4);
        initial_state_matrix.col(0) = initial_state_.position;
        initial_state_matrix.col(1) = initial_state_.velocity;
        initial_state_matrix.col(2) = initial_state_.acceleration;
        initial_state_matrix.col(3) = initial_state_.jerk;

        Eigen::MatrixXd final_state_matrix(3, 4);
        final_state_matrix.col(0) = final_state.position;
        final_state_matrix.col(1) = final_state.velocity;
        final_state_matrix.col(2) = final_state.acceleration;
        final_state_matrix.col(3) = final_state.jerk;

        minco_optimizer_.generate(initial_state_matrix, final_state_matrix, intermediate_waypoints, piece_duration);
        double cost = minco_optimizer_.getTrajSnapCost();

        addTimeIntegralPenalty(cost);

        minco_optimizer_.calGrads_CT();
        minco_optimizer_.calGrads_PT();

        // Backpropagate gradient from final state's dependency on time
        minco_optimizer_.gdT += minco_optimizer_.gdTail.col(0).dot(target_traj_->getVelocity(total_duration));
        minco_optimizer_.gdT += minco_optimizer_.gdTail.col(1).dot(target_traj_->getAcceleration(total_duration));
        // We assume target jerk is zero, so no gradient contribution from final acceleration.

        // Add time regularization cost and gradient
        cost += params_.time_weight * total_duration;
        minco_optimizer_.gdT += params_.time_weight * params_.traj_pieces_num;

        // Populate gradient array for L-BFGS
        grads[0] = minco_optimizer_.gdT * gradTimeTransform(log_time_var);
        Eigen::Map<Eigen::MatrixXd> grad_intermediate_waypoints(grads + params_.time_var_dim, 3, params_.waypoint_num);
        grad_intermediate_waypoints = minco_optimizer_.gdP;

        return cost;
    }

   private:
    // Static wrapper for the L-BFGS callback
    static double _objectiveFunction(void* ptr, const double* vars, double* grads, int n) {
        SimpleCatching* optimizer = reinterpret_cast<SimpleCatching*>(ptr);
        return optimizer->objectiveFunction(ptr, vars, grads, n);
    }

    std::shared_ptr<SimpleTrajectory> target_traj_;
    std::shared_ptr<Trajectory> initial_guess_traj_;
    Eigen::Quaterniond catching_att_{1.0, 0.0, 0.0, 0.0};
    bool has_initial_guess_;
};

#endif  // SIMPLE_CATCHING_H
