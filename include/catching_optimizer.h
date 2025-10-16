#ifndef CATCHING_OPTIMIZER_H
#define CATCHING_OPTIMIZER_H

#include <Eigen/Eigen>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

#ifndef NDEBUG
#include <iostream>
#endif

#include "lbfgs_raw.hpp"
#include "minco.hpp"
#include "poly_traj_utils.hpp"
#include "simple_trajectory.h"

struct DroneState {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Vector3d acceleration;
    Eigen::Vector3d jerk;
    Eigen::Quaterniond attitude_quat;
    Eigen::Vector3d attitude;
    Eigen::Vector3d body_rate;

    DroneState() {
        position.setZero();
        velocity.setZero();
        acceleration.setZero();
        jerk.setZero();
        attitude.setZero();
        body_rate.setZero();
        attitude_quat.setIdentity();
    }

    DroneState(const Eigen::Vector3d& pos, const Eigen::Vector3d& vel, const Eigen::Vector3d& acc) {
        position = pos;
        velocity = vel;
        acceleration = acc;
        jerk.setZero();
        attitude.setZero();
        body_rate.setZero();
        attitude_quat.setIdentity();
    }
};

struct TrajOptParameters {
    // Dynamic Limits
    double max_velocity = 10.0;
    double max_acceleration = 10.0;
    double max_thrust = 20.0;
    double min_thrust = 2.0;
    double max_body_rate = 3.0;
    double max_yaw_rate = 2.0;
    double thrust_half_level = -1.0;
    double thrust_half_range = -1.0;
    double min_z = 0.4;

    // Optimization Weights
    double time_weight = 1.0;
    double pos_penalty_weight = 100.0;
    double vel_penalty_weight = 10.0;
    double acc_penalty_weight = 1.0;
    double thrust_weight = 1.0;
    double body_rate_weight = 1.0;
    double collision_weight = 1.0;
    double terminal_pos_weight = 1.0;
    double terminal_vel_weight = 1.0;
    double terminal_acc_weight = 1.0;
    double terminal_att_weight = 1.0;

    // Numerical Parameters
    int integration_steps = 20;
    int traj_pieces_num = 1;
    int time_var_dim = 1;
    int waypoint_num = 1;
    int custom_var_dim = 3;

    // Environmental Parameters
    Eigen::Vector3d gravity_vec = Eigen::Vector3d(0.0, 0.0, -9.81);

    // LBFGS Parameters
    int lbfgs_mem_size = 32;
    int lbfgs_past = 3;
    double lbfgs_g_epsilon = 0.0;
    double lbfgs_min_step = 1e-16;
    double lbfgs_delta = 1e-4;
    int lbfgs_line_search_type = 0;
};

class CatchingOptimizer {
   public:
    CatchingOptimizer()
        : optimization_vars_(nullptr),
          iteration_count_(0),
          log_time_var_(0.0),
          optimized_total_duration_(0.0),
          inner_loop_duration_(0.0),
          integral_duration_(0.0) {
        initial_state_matrix_.resize(3, 4);
        initial_state_matrix_.setZero();
    }
    ~CatchingOptimizer() {
        if (optimization_vars_ != nullptr) {
            delete[] optimization_vars_;
            optimization_vars_ = nullptr;
        }
    }

    bool generateTrajectory(Trajectory& trajectory) {
        if (!initial_state_set_ || !terminal_state_set_ || !target_traj_set_ || !target_traj_) {
            throw std::runtime_error("[CatchingOptimizer] Optimizer not fully configured. Set initial, terminal, and target states.");
        }

        assert(params_.custom_var_dim == 0 && "Custom variable dimension must be 0 for catching.");
        const int variable_count = params_.time_var_dim + 3 * params_.waypoint_num;

        if (optimization_vars_ != nullptr) {
            delete[] optimization_vars_;
        }
        optimization_vars_ = new double[variable_count];

        double& log_time_var = optimization_vars_[0];
        Eigen::Map<Eigen::MatrixXd>
            intermediate_waypoints(optimization_vars_ + params_.time_var_dim, 3, params_.waypoint_num);

        minco_optimizer_.reset(params_.traj_pieces_num);

        double estimated_duration = 0.0;
        CoefficientMat coefficient_matrix;
        const double target_traj_duration = target_traj_->getTotalDuration();

        if (initial_guess_set_) {
            estimated_duration = initial_guess_duration_;
            solveBoundaryValueProblem(estimated_duration, initial_state_, desired_terminal_state_, coefficient_matrix);
        } else {
            double min_duration = (target_traj_->getPosition(0.0) - initial_state_.position).norm() / params_.max_velocity;
            estimated_duration = min_duration;
            double max_body_rate = 0.0;
            do {
                estimated_duration += 0.2;
                double boundary_sample_time = std::min(estimated_duration, target_traj_duration);
                
                DroneState bvp_terminal_state;
                bvp_terminal_state.position = target_traj_->getPosition(boundary_sample_time);
                bvp_terminal_state.velocity = target_traj_->getVelocity(boundary_sample_time);
                bvp_terminal_state.acceleration = -params_.gravity_vec;
                bvp_terminal_state.jerk.setZero();

                solveBoundaryValueProblem(estimated_duration, initial_state_, bvp_terminal_state, coefficient_matrix);
                Trajectory boundary_traj({estimated_duration}, {coefficient_matrix});
                max_body_rate = getMaxBodyRate(boundary_traj);
            } while (max_body_rate > 1.5 * params_.max_body_rate);
        }

        std::cout << "\033[33m" << "[CatchingOptimizer] Initial estimated duration: " << estimated_duration << "\033[0m" << std::endl;

        Eigen::VectorXd poly_terms(8);
        poly_terms(7) = 1.0;
        for (int i = 0; i < params_.waypoint_num; ++i) {
            double segment_time = (static_cast<double>(i + 1) / params_.traj_pieces_num) * estimated_duration;
            for (int j = 6; j >= 0; --j) {
                poly_terms(j) = poly_terms(j + 1) * segment_time;
            }
            intermediate_waypoints.col(i) = coefficient_matrix * poly_terms;
        }
        log_time_var = logC2(estimated_duration / params_.traj_pieces_num);

        lbfgs::lbfgs_parameter_t lbfgs_params{};
        lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
        lbfgs_params.mem_size = params_.lbfgs_mem_size;
        lbfgs_params.past = params_.lbfgs_past;
        lbfgs_params.g_epsilon = params_.lbfgs_g_epsilon;
        lbfgs_params.min_step = params_.lbfgs_min_step;
        lbfgs_params.delta = params_.lbfgs_delta;
        lbfgs_params.line_search_type = params_.lbfgs_line_search_type;

        double min_objective = 0.0;
        iteration_count_ = 0;
        int optimization_result = lbfgs::lbfgs_optimize(
            variable_count, optimization_vars_, &min_objective,
            &CatchingOptimizer::objectiveFunction, nullptr, nullptr, this, &lbfgs_params);

        if (optimization_result < 0) {
            delete[] optimization_vars_;
            optimization_vars_ = nullptr;
            return false;
        }

        double piece_duration = expC2(log_time_var);
        double total_duration = params_.traj_pieces_num * piece_duration;
        double final_sample_time = std::min(total_duration, target_traj_duration);
        Eigen::Vector3d final_target_pos = target_traj_->getPosition(final_sample_time);

        Eigen::MatrixXd terminal_state_matrix(3, 4);
        terminal_state_matrix.setZero();
        terminal_state_matrix.col(0) = final_target_pos;

        minco_optimizer_.generate(initial_state_matrix_, terminal_state_matrix, intermediate_waypoints, piece_duration);
        trajectory = minco_optimizer_.getTraj();

        delete[] optimization_vars_;
        optimization_vars_ = nullptr;
        return true;
    }

   private:
    /* states */
    DroneState initial_state_;
    Eigen::MatrixXd initial_state_matrix_;
    DroneState desired_terminal_state_;

    /* optimizer and params */
    TrajOptParameters params_;
    minco::MINCO_S4_Uniform minco_optimizer_;
    lbfgs::lbfgs_parameter_t lbfgs_params_;

    /* optimization vars */
    double* optimization_vars_ = nullptr;
    int iteration_count_ = 0;
    double log_time_var_ = 0.0;
    double optimized_total_duration_ = 0.0;
    double inner_loop_duration_ = 0.0;
    double integral_duration_ = 0.0;

    /* target and initial guess */
    std::shared_ptr<SimpleTrajectory> target_traj_;
    bool initial_state_set_ = false;
    bool terminal_state_set_ = false;
    bool target_traj_set_ = false;
    bool initial_guess_set_ = false;
    double initial_guess_duration_ = 0.0;
    Eigen::Vector3d initial_guess_pursuer_intercept_pos_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d initial_guess_pursuer_intercept_vel_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d initial_guess_pursuer_intercept_acc_ = Eigen::Vector3d::Zero();

    /* terminal catching attitude */
    Eigen::Vector3d catching_att_z_vec_ = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d catching_basis_x_ = Eigen::Vector3d::UnitX();
    Eigen::Vector3d catching_basis_y_ = Eigen::Vector3d::UnitY();
    Eigen::Quaterniond catching_att_ = Eigen::Quaterniond::Identity();

    void addTimeIntegralPenalty(double& cost) {
        Eigen::Vector3d pos, vel, acc, jer, snap;
        Eigen::Vector3d grad_temp_pos, grad_temp_vel, grad_temp_acc, grad_temp_jer;
        Eigen::Vector3d grad_pos_total, grad_vel_total, grad_acc_total, grad_jer_total;
        double cost_temp = 0.0;
        double cost_inner = 0.0;

        Eigen::Matrix<double, 8, 1> beta0, beta1, beta2, beta3, beta4;
        double sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7;
        double step = 0.0;
        double alpha = 0.0;
        Eigen::Matrix<double, 8, 3> grad_coeff;
        double grad_time = 0.0;
        double integration_weight = 0.0;

        int inner_loop = params_.integration_steps + 1;
        step = minco_optimizer_.t(1) / params_.integration_steps;

        sigma1 = 0.0;

        for (int j = 0; j < inner_loop; ++j) {
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
            alpha = 1.0 / params_.integration_steps * j;
            integration_weight = (j == 0 || j == inner_loop - 1) ? 0.5 : 1.0;

            for (int i = 0; i < params_.traj_pieces_num; ++i) {
                const auto& coeff_block = minco_optimizer_.c.block<8, 3>(i * 8, 0);

                pos = coeff_block.transpose() * beta0;
                vel = coeff_block.transpose() * beta1;
                acc = coeff_block.transpose() * beta2;
                jer = coeff_block.transpose() * beta3;
                snap = coeff_block.transpose() * beta4;

                grad_pos_total.setZero();
                grad_vel_total.setZero();
                grad_acc_total.setZero();
                grad_jer_total.setZero();
                cost_inner = 0.0;

                if (computeFloorCost(pos, grad_temp_pos, cost_temp)) {
                    grad_pos_total += grad_temp_pos;
                    cost_inner += cost_temp;
                }

                if (computeVelocityCost(vel, grad_temp_vel, cost_temp)) {
                    grad_vel_total += grad_temp_vel;
                    cost_inner += cost_temp;
                }

                if (computeThrustCost(acc, grad_temp_acc, cost_temp)) {
                    grad_acc_total += grad_temp_acc;
                    cost_inner += cost_temp;
                }

                if (computeBodyRateCost(acc, jer, grad_temp_acc, grad_temp_jer, cost_temp)) {
                    grad_acc_total += grad_temp_acc;
                    grad_jer_total += grad_temp_jer;
                    cost_inner += cost_temp;
                }

                if (computeBodyRateYawCost(acc, jer, grad_temp_acc, grad_temp_jer, cost_temp)) {
                    grad_acc_total += grad_temp_acc;
                    grad_jer_total += grad_temp_jer;
                    cost_inner += cost_temp;
                }

                grad_coeff = beta0 * grad_pos_total.transpose();
                grad_time = grad_pos_total.transpose() * vel;
                grad_coeff += beta1 * grad_vel_total.transpose();
                grad_time += grad_vel_total.transpose() * acc;
                grad_coeff += beta2 * grad_acc_total.transpose();
                grad_time += grad_acc_total.transpose() * jer;
                grad_coeff += beta3 * grad_jer_total.transpose();
                grad_time += grad_jer_total.transpose() * snap;

                minco_optimizer_.gdC.block<8, 3>(i * 8, 0) += integration_weight * step * grad_coeff;
                minco_optimizer_.gdT += integration_weight * (cost_inner / params_.integration_steps + alpha * step * grad_time);
                cost += integration_weight * step * cost_inner;
            }
            sigma1 += step;
        }
    }

    bool computeVelocityCost(const Eigen::Vector3d& velocity, Eigen::Vector3d& grad_velocity,
                             double& cost_velocity) const {
        double velocity_penalty = velocity.squaredNorm() - params_.max_velocity * params_.max_velocity;
        if (velocity_penalty > 0.0) {
            double gradient = 0.0;
            cost_velocity = smoothedL1(velocity_penalty, gradient);
            grad_velocity = params_.vel_penalty_weight * gradient * 2.0 * velocity;
            cost_velocity *= params_.vel_penalty_weight;
            return true;
        }
        return false;
    }

    bool computeThrustCost(const Eigen::Vector3d& acceleration,
                           Eigen::Vector3d& grad_acceleration,
                           double& cost_acceleration) {
        bool has_penalty = false;
        grad_acceleration.setZero();
        cost_acceleration = 0.0;
        Eigen::Vector3d thrust = acceleration - params_.gravity_vec;

        auto max_penalty = thrust.squaredNorm() - params_.max_thrust * params_.max_thrust;
        if (max_penalty > 0.0) {
            double gradient = 0.0;
            cost_acceleration = params_.thrust_weight * smoothedL1(max_penalty, gradient);
            grad_acceleration = params_.thrust_weight * 2.0 * gradient * thrust;
            has_penalty = true;
        }

        double min_penalty = params_.min_thrust * params_.min_thrust - thrust.squaredNorm();
        if (min_penalty > 0.0) {
            double gradient = 0.0;
            cost_acceleration = params_.thrust_weight * smoothedL1(min_penalty, gradient);
            grad_acceleration = -params_.thrust_weight * 2.0 * gradient * thrust;
            has_penalty = true;
        }

        return has_penalty;
    }

    bool computeBodyRateCost(const Eigen::Vector3d& acceleration,
                             const Eigen::Vector3d& jerk,
                             Eigen::Vector3d& grad_acceleration,
                             Eigen::Vector3d& grad_jerk,
                             double& cost) {
        Eigen::Vector3d thrust = acceleration - params_.gravity_vec;
        Eigen::Vector3d zb_dot = getNormalizationJacobian(thrust) * jerk;
        double body_rate_sq = zb_dot.squaredNorm();
        double penalty = body_rate_sq - params_.max_body_rate * params_.max_body_rate;
        if (penalty > 0.0) {
            double gradient = 0.0;
            cost = smoothedL1(penalty, gradient);

            Eigen::Vector3d grad_zb_dot = 2.0 * zb_dot;
            grad_jerk = getNormalizationJacobian(thrust).transpose() * grad_zb_dot;
            grad_acceleration = getNormalizationHessian(thrust, jerk).transpose() * grad_zb_dot;

            cost *= params_.body_rate_weight;
            gradient *= params_.body_rate_weight;
            grad_acceleration *= gradient;
            grad_jerk *= gradient;

            return true;
        }
        return false;
    }

    static bool computeBodyRateYawCost(const Eigen::Vector3d& acceleration,
                                       const Eigen::Vector3d& jerk,
                                       Eigen::Vector3d& grad_acceleration,
                                       Eigen::Vector3d& grad_jerk,
                                       double& cost) {
        (void)acceleration;
        (void)jerk;
        (void)grad_acceleration;
        (void)grad_jerk;
        (void)cost;
        return false;
    }

    bool computeFloorCost(const Eigen::Vector3d& position,
                          Eigen::Vector3d& grad_position,
                          double& cost_position) const {
        double penalty = params_.min_z - position.z();
        if (penalty > 0.0) {
            double gradient = 0.0;
            cost_position = smoothedL1(penalty, gradient);
            cost_position *= params_.pos_penalty_weight;
            grad_position.setZero();
            grad_position.z() = -params_.pos_penalty_weight * gradient;
            return true;
        }
        return false;
    }

    Eigen::Map<Eigen::MatrixXd>
    getIntermediateWaypoints(const Eigen::Vector3d& start_pos, const Eigen::Vector3d& end_pos,
                             int num_pieces, bool use_straight_line = true,
                             const Trajectory& trajectory = getDefaultTrajectory()) {
        Eigen::Map<Eigen::MatrixXd> waypoints(optimization_vars_ + params_.time_var_dim, 3, params_.waypoint_num);

        if (!use_straight_line && trajectory.getPieceNum() > 0) {
            double total_duration = trajectory.getTotalDuration();
            for (int i = 0; i < params_.waypoint_num; ++i) {
                double ratio = static_cast<double>(i + 1) / static_cast<double>(num_pieces);
                double sample_time = ratio * total_duration;
                waypoints.col(i) = trajectory.getPos(sample_time);
            }
        } else {
            for (int i = 0; i < params_.waypoint_num; ++i) {
                double ratio = static_cast<double>(i + 1) / static_cast<double>(num_pieces);
                waypoints.col(i) = start_pos + ratio * (end_pos - start_pos);
            }
        }

        return waypoints;
    }

    double getMaxBodyRate(Trajectory& trajectory) {
        double dt = 0.01;
        double max_body_rate = 0.0;
        for (double t = 0.0; t < trajectory.getTotalDuration(); t += dt) {
            Eigen::Vector3d acc = trajectory.getAcc(t);
            Eigen::Vector3d jer = trajectory.getJer(t);
            Eigen::Vector3d thrust = acc - params_.gravity_vec;
            Eigen::Vector3d zb_dot = getNormalizationJacobian(thrust) * jer;
            double body_rate_12 = zb_dot.norm();
            if (body_rate_12 > max_body_rate) {
                max_body_rate = body_rate_12;
            }
        }
        return max_body_rate;
    }

    bool computePerchingCollisionCost(const Eigen::Vector3d& position,
                                      const Eigen::Vector3d& acceleration,
                                      const Eigen::Vector3d& target_position,
                                      Eigen::Vector3d& grad_position,
                                      Eigen::Vector3d& grad_acceleration,
                                      Eigen::Vector3d& grad_target_position,
                                      double& cost) {
        (void)position;
        (void)acceleration;
        (void)target_position;
        (void)grad_position;
        (void)grad_acceleration;
        (void)grad_target_position;
        (void)cost;
        return false;
    }

    static double objectiveFunction(void* ptr_optimizer, const double* vars, double* grads, int n) {
        (void)n;
        auto* optimizer = static_cast<CatchingOptimizer*>(ptr_optimizer);
        optimizer->iteration_count_++;

        const double& log_time_var = vars[0];
        Eigen::Map<const Eigen::MatrixXd> intermediate_waypoints(
            vars + optimizer->params_.time_var_dim, 3, optimizer->params_.waypoint_num);

        double piece_duration = expC2(log_time_var);
        double total_duration = optimizer->params_.traj_pieces_num * piece_duration;

        double final_sample_time = total_duration;
        double target_traj_duration = optimizer->target_traj_->getTotalDuration();
        if (target_traj_duration > 0.0) {
            final_sample_time = std::min(std::max(final_sample_time, 0.0), target_traj_duration);
        }
        Eigen::Vector3d final_pos = optimizer->target_traj_->getPosition(final_sample_time);

        Eigen::MatrixXd terminal_state_matrix(3, 4);
        terminal_state_matrix.setZero();
        terminal_state_matrix.col(0) = final_pos;

        optimizer->minco_optimizer_.generate(
            optimizer->initial_state_matrix_, terminal_state_matrix,
            intermediate_waypoints, piece_duration);

        double cost = optimizer->minco_optimizer_.getTrajSnapCost();
        optimizer->minco_optimizer_.calGrads_CT();
        optimizer->addTimeIntegralPenalty(cost);

        const Trajectory& current_traj = optimizer->minco_optimizer_.getTraj();
        Eigen::Vector3d actual_final_acc = current_traj.getJuncAcc(optimizer->params_.traj_pieces_num);
        Eigen::Vector3d required_thrust = actual_final_acc - optimizer->params_.gravity_vec;
        Eigen::Vector3d desired_thrust_dir = optimizer->desired_terminal_state_.attitude_quat * Eigen::Vector3d::UnitZ();
        Eigen::Vector3d thrust_dir_error = required_thrust.normalized() - desired_thrust_dir;
        cost += optimizer->params_.terminal_att_weight * thrust_dir_error.squaredNorm();
        
        optimizer->minco_optimizer_.calGrads_PT();

        optimizer->minco_optimizer_.gdT += optimizer->minco_optimizer_.gdTail.col(0).dot(
            optimizer->target_traj_->getVelocity(final_sample_time));
        
        Eigen::Vector3d grad_att_wrt_acc = getNormalizationJacobian(required_thrust).transpose() *
                                           (optimizer->params_.terminal_att_weight * 2.0 * thrust_dir_error);
        optimizer->minco_optimizer_.gdTail.col(2) += grad_att_wrt_acc;

        cost += optimizer->params_.time_weight * total_duration;
        optimizer->minco_optimizer_.gdT += optimizer->params_.time_weight * optimizer->params_.traj_pieces_num;

        grads[0] = optimizer->minco_optimizer_.gdT * gradTimeTransform(log_time_var);
        Eigen::Map<Eigen::MatrixXd> grad_intermediate_waypoints(
            grads + optimizer->params_.time_var_dim, 3, optimizer->params_.waypoint_num);
        grad_intermediate_waypoints = optimizer->minco_optimizer_.gdP;

        return cost;
    }

    void updateTerminalState(const Eigen::Vector3d& target_pos, const Eigen::Vector3d& target_vel) {
        // update terminal state based on my rule
        // now the rule is to match the pos and vel directly

        // TODO: update this later
        // For agent: do not edit this until I allow you to
        desired_terminal_state_.position = target_pos;
        // desired_terminal_state_.velocity = target_vel;

        Eigen::Vector3d pos2target_dir = (target_pos - initial_state_.position);
        pos2target_dir = normalizeVector(pos2target_dir);
        desired_terminal_state_.velocity = pos2target_dir * 0.5 * params_.max_velocity;

        desired_terminal_state_.acceleration.setZero();
        desired_terminal_state_.jerk.setZero();
    }

    /* Static private helper functions */
    static void solveBoundaryValueProblem(double duration,
                                          const DroneState& initial_state,
                                          const DroneState& final_state,
                                          CoefficientMat& coefficient_matrix) {
        const double t1 = duration;
        const double t2 = t1 * t1;
        const double t3 = t2 * t1;
        const double t4 = t2 * t2;
        const double t5 = t3 * t2;
        const double t6 = t3 * t3;
        const double t7 = t4 * t3;

        CoefficientMat boundary_cond;
        // Set up boundary conditions matrix
        boundary_cond.col(0) = initial_state.position;
        boundary_cond.col(1) = initial_state.velocity;
        boundary_cond.col(2) = initial_state.acceleration;
        boundary_cond.col(3) = initial_state.jerk;
        boundary_cond.col(4) = final_state.position;
        boundary_cond.col(5) = final_state.velocity;
        boundary_cond.col(6) = final_state.acceleration;
        boundary_cond.col(7) = final_state.jerk;

        // Solve for polynomial coefficients
        coefficient_matrix.col(0) =
            (boundary_cond.col(7) / 6.0 + boundary_cond.col(3) / 6.0) * t3 +
            (-2.0 * boundary_cond.col(6) + 2.0 * boundary_cond.col(2)) * t2 +
            (10.0 * boundary_cond.col(5) + 10.0 * boundary_cond.col(1)) * t1 +
            (-20.0 * boundary_cond.col(4) + 20.0 * boundary_cond.col(0));
        coefficient_matrix.col(1) =
            (-0.5 * boundary_cond.col(7) - boundary_cond.col(3) / 1.5) * t3 +
            (6.5 * boundary_cond.col(6) - 7.5 * boundary_cond.col(2)) * t2 +
            (-34.0 * boundary_cond.col(5) - 36.0 * boundary_cond.col(1)) * t1 +
            (70.0 * boundary_cond.col(4) - 70.0 * boundary_cond.col(0));
        coefficient_matrix.col(2) =
            (0.5 * boundary_cond.col(7) + boundary_cond.col(3)) * t3 +
            (-7.0 * boundary_cond.col(6) + 10.0 * boundary_cond.col(2)) * t2 +
            (39.0 * boundary_cond.col(5) + 45.0 * boundary_cond.col(1)) * t1 +
            (-84.0 * boundary_cond.col(4) + 84.0 * boundary_cond.col(0));
        coefficient_matrix.col(3) =
            (-boundary_cond.col(7) / 6.0 - boundary_cond.col(3) / 1.5) * t3 +
            (2.5 * boundary_cond.col(6) - 5.0 * boundary_cond.col(2)) * t2 +
            (-15.0 * boundary_cond.col(5) - 20.0 * boundary_cond.col(1)) * t1 +
            (35.0 * boundary_cond.col(4) - 35.0 * boundary_cond.col(0));
        coefficient_matrix.col(4) = boundary_cond.col(3) / 6.0;
        coefficient_matrix.col(5) = boundary_cond.col(2) / 2.0;
        coefficient_matrix.col(6) = boundary_cond.col(1);
        coefficient_matrix.col(7) = boundary_cond.col(0);

        coefficient_matrix.col(0) = coefficient_matrix.col(0) / t7;
        coefficient_matrix.col(1) = coefficient_matrix.col(1) / t6;
        coefficient_matrix.col(2) = coefficient_matrix.col(2) / t5;
        coefficient_matrix.col(3) = coefficient_matrix.col(3) / t4;
    }

    static bool quaternionToZAxis(const Eigen::Quaterniond& quat, Eigen::Vector3d& z_axis) {
        Eigen::Matrix3d rotation = quat.toRotationMatrix();
        z_axis = rotation.col(2);
        return true;
    }

    static Eigen::Vector3d normalizeVector(const Eigen::Vector3d& vec) {
        return vec.normalized();
    }

    static Eigen::Matrix3d getNormalizationJacobian(const Eigen::Vector3d& vec) {
        double norm_sq = vec.squaredNorm();
        return (Eigen::Matrix3d::Identity() - vec * vec.transpose() / norm_sq) / std::sqrt(norm_sq);
    }

    static Eigen::Matrix3d getNormalizationHessian(const Eigen::Vector3d& vec,
                                                   const Eigen::Vector3d& dir) {
        double norm_sq = vec.squaredNorm();
        double norm_cu = norm_sq * vec.norm();
        Eigen::Matrix3d term = (3.0 * vec * vec.transpose() / norm_sq - Eigen::Matrix3d::Identity());
        return (term * dir * vec.transpose() - vec * dir.transpose() -
                vec.dot(dir) * Eigen::Matrix3d::Identity()) /
               norm_cu;
    }

    static double smoothedL1(const double& value, double& gradient) {
        static double mu = 0.01;
        if (value < 0.0) {
            return 0.0;
        } else if (value > mu) {
            gradient = 1.0;
            return value - 0.5 * mu;
        } else {
            const double ratio = value / mu;
            const double ratio_sq = ratio * ratio;
            const double mu_minus_half_value = mu - 0.5 * value;
            gradient = ratio_sq * ((-0.5) * ratio + 3.0 * mu_minus_half_value / mu);
            return mu_minus_half_value * ratio_sq * ratio;
        }
    }

    static double smoothedZeroOne(const double& value, double& gradient) {
        static double mu = 0.01;
        static double mu4 = mu * mu * mu * mu;
        static double inv_mu4 = 1.0 / mu4;
        if (value < -mu) {
            gradient = 0.0;
            return 0.0;
        } else if (value < 0.0) {
            double y = value + mu;
            double y2 = y * y;
            gradient = y2 * (mu - 2.0 * value) * inv_mu4;
            return 0.5 * y2 * y * (mu - value) * inv_mu4;
        } else if (value < mu) {
            double y = value - mu;
            double y2 = y * y;
            gradient = y2 * (mu + 2.0 * value) * inv_mu4;
            return 0.5 * y2 * y * (mu + value) * inv_mu4 + 1.0;
        } else {
            gradient = 0.0;
            return 1.0;
        }
    }

    static double expC2(double time_var) {
        return time_var > 0.0 ? ((0.5 * time_var + 1.0) * time_var + 1.0)
                              : 1.0 / ((0.5 * time_var - 1.0) * time_var + 1.0);
    }

    static double logC2(double duration) {
        return duration > 1.0 ? (std::sqrt(2.0 * duration - 1.0) - 1.0)
                              : (1.0 - std::sqrt(2.0 / duration - 1.0));
    }

    static double gradTimeTransform(double time_var) {
        if (time_var > 0.0) {
            return time_var + 1.0;
        } else {
            double denom = (0.5 * time_var - 1.0) * time_var + 1.0;
            return (1.0 - time_var) / (denom * denom);
        }
    }

    static double forwardThrust(double tail_angle, double thrust_half_range, double thrust_mid_level) {
        return thrust_half_range * std::sin(tail_angle) + thrust_mid_level;
    }

    static double propagateThrustGradient(double tail_angle, double thrust_gradient, double thrust_half_range) {
        return thrust_half_range * std::cos(tail_angle) * thrust_gradient;
    }

    static void computeTailVelocity(const Eigen::Vector2d& tail_velocity_params,
                                    const Eigen::Vector3d& landing_velocity,
                                    const Eigen::Vector3d& basis_x,
                                    const Eigen::Vector3d& basis_y,
                                    Eigen::Vector3d& tail_velocity) {
        tail_velocity = landing_velocity + tail_velocity_params.x() * basis_x + tail_velocity_params.y() * basis_y;
    }

    static const Trajectory& getDefaultTrajectory() {
        static Trajectory default_trajectory;
        return default_trajectory;
    }

    static int earlyExitCallback(void* ptr_optimizer,
                                 const double* vars,
                                 const double* grads,
                                 const double fx,
                                 const double xnorm,
                                 const double gnorm,
                                 const double step,
                                 int n,
                                 int k,
                                 int ls) {
        (void)ptr_optimizer;
        (void)vars;
        (void)grads;
        (void)fx;
        (void)xnorm;
        (void)gnorm;
        (void)step;
        (void)n;
        (void)k;
        (void)ls;
        return 0;
    }

   public:
    /* Public helper functions */
    void setDynamicLimits(double max_velocity = 10.0,
                          double max_acceleration = 10.0,
                          double max_thrust = 20.0,
                          double min_thrust = 2.0,
                          double max_body_rate = 3.0,
                          double max_yaw_body_rate = 2.0) {
        assert(max_velocity > 0.0 && "Max velocity must be positive");
        assert(max_acceleration > 0.0 && "Max acceleration must be positive");
        assert(max_thrust > min_thrust && "Max thrust must be greater than min thrust");
        assert(max_body_rate > 0.0 && "Max body rate must be positive");
        assert(max_yaw_body_rate > 0.0 && "Max yaw body rate must be positive");

        params_.max_velocity = max_velocity;
        params_.max_acceleration = max_acceleration;
        params_.max_thrust = max_thrust;
        params_.min_thrust = min_thrust;
        params_.max_body_rate = max_body_rate;
        params_.max_yaw_rate = max_yaw_body_rate;

        params_.thrust_half_level = 0.5 * (max_thrust + min_thrust);
        params_.thrust_half_range = 0.5 * (max_thrust - min_thrust);
    }

    void setOptimizationWeights(double time_weight = 1.0,
                                double position_weight = 1.0,
                                double velocity_weight = 1.0,
                                double acceleration_weight = 1.0,
                                double thrust_weight = 1.0,
                                double body_rate_weight = 1.0,
                                double terminal_velocity_weight = -1.0,
                                double collision_weight = 1.0) {
        assert(time_weight >= 0.0 && "Time weight must be non-negative");
        assert(position_weight >= 0.0 && "Position weight must be non-negative");
        assert(velocity_weight >= 0.0 && "Velocity weight must be non-negative");
        assert(acceleration_weight >= 0.0 && "Acceleration weight must be non-negative");
        assert(thrust_weight >= 0.0 && "Thrust weight must be non-negative");
        assert(body_rate_weight >= 0.0 && "Body rate weight must be non-negative");
        assert(collision_weight >= 0.0 && "Perching collision weight must be non-negative");

        params_.time_weight = time_weight;
        params_.pos_penalty_weight = position_weight;
        params_.vel_penalty_weight = velocity_weight;
        params_.acc_penalty_weight = acceleration_weight;
        params_.thrust_weight = thrust_weight;
        params_.body_rate_weight = body_rate_weight;
        params_.collision_weight = collision_weight;
        params_.terminal_vel_weight = terminal_velocity_weight;
    }

    void setTrajectoryParams(int integration_steps = 20,
                             int traj_pieces_num = 1,
                             int time_var_dim = 1,
                             int custom_var_num = 3) {
        params_.integration_steps = integration_steps;
        params_.traj_pieces_num = traj_pieces_num;
        params_.time_var_dim = time_var_dim;
        params_.custom_var_dim = custom_var_num;
        params_.waypoint_num = std::max(0, traj_pieces_num - 1);
    }

    void setInitialState(const DroneState& initial_state) {
        initial_state_ = initial_state;
        if (initial_state_matrix_.rows() != 3 || initial_state_matrix_.cols() != 4) {
            initial_state_matrix_.resize(3, 4);
        }
        initial_state_matrix_.col(0) = initial_state_.position;
        initial_state_matrix_.col(1) = initial_state_.velocity;
        initial_state_matrix_.col(2) = initial_state_.acceleration;
        initial_state_matrix_.col(3) = initial_state_.jerk;

        initial_state_set_ = true;
    }

    void setTerminalState(const DroneState& final_state) {
        desired_terminal_state_ = final_state;
        terminal_state_set_ = true;
    }

    int getIterationCount() const { return iteration_count_; }
    Trajectory getCurrentTrajectory() const { return minco_optimizer_.getTraj(); }

    static Eigen::Quaterniond euler2Quaternion(const Eigen::Vector3d& euler) {
        return Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX());
    }

    static void q2EulerAngle(const Eigen::Quaterniond& q, Eigen::Vector3d& euler) {
        double sr_cp = 2.0 * (q.w() * q.x() + q.y() * q.z());
        double cr_cp = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
        euler[0] = atan2(sr_cp, cr_cp);

        double sin_p = 2.0 * (q.w() * q.y() - q.z() * q.x());
        if (fabs(sin_p) >= 1)
            euler[1] = copysign(M_PI / 2, sin_p);  // pi/2
        else
            euler[1] = asin(sin_p);

        double sy_cp = 2.0 * (q.w() * q.z() + q.x() * q.y());
        double cy_cp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
        euler[2] = atan2(sy_cp, cy_cp);
    }

    static Eigen::Matrix3d rotB2A(const Eigen::Vector3d& att) {
        double phi = att[0];
        double theta = att[1];

        Eigen::Matrix3d rotation_matrix;
        rotation_matrix << 1, tan(theta) * sin(phi), tan(theta) * cos(phi),
            0, cos(phi), -sin(phi),
            0, sin(phi) / (cos(theta) + 1e-8), cos(phi) / (cos(theta) + 1e-8);

        return rotation_matrix;
    }

    static Eigen::Matrix3d rotB2ody2World(const Eigen::Vector3d& att) {
        double s_phi = sin(att[0]);
        double c_phi = cos(att[0]);
        double s_theta = sin(att[1]);
        double c_theta = cos(att[1]);
        double s_psi = sin(att[2]);
        double c_psi = cos(att[2]);

        Eigen::Matrix3d rot_mat;
        rot_mat << c_theta * c_psi, s_theta * s_phi * c_psi - s_psi * c_phi, s_theta * c_phi * c_psi + s_psi * s_phi,
            c_theta * s_psi, s_psi * s_theta * s_phi + c_psi * c_phi, s_psi * s_theta * c_phi - c_psi * s_phi,
            -s_theta, s_phi * c_theta, c_phi * c_theta;

        return rot_mat;
    }

    void setTargetTrajectory(std::shared_ptr<SimpleTrajectory> target_trajectory) {
        target_traj_ = target_trajectory;
        target_traj_set_ = true;
    }

    void setCatchingAttitude(const Eigen::Vector3d& euler_attitude) {
        // call setCatchingAttitude or set directly in setTerminalState
        desired_terminal_state_.attitude = euler_attitude;
        desired_terminal_state_.attitude_quat = euler2Quaternion(euler_attitude);
    }

    void setCatchingAttitude(const Eigen::Quaterniond& quat_attitude) {
        q2EulerAngle(quat_attitude, desired_terminal_state_.attitude);
        desired_terminal_state_.attitude_quat = quat_attitude;
    }

    void setInitialGuess(const double intercept_time, const Eigen::Vector3d& intercept_pos,
                         const Eigen::Vector3d& intercept_vel, const Eigen::Vector3d& intercept_acc = Eigen::Vector3d::Zero()) {
        initial_guess_set_ = true;

        initial_guess_duration_ = intercept_time;
        initial_guess_pursuer_intercept_pos_ = intercept_pos;
        initial_guess_pursuer_intercept_vel_ = intercept_vel;
        initial_guess_pursuer_intercept_acc_ = intercept_acc;
    }
};

#endif /* CATCHING_OPTIMIZER_H */
