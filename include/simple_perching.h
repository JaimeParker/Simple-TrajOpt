#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cassert>
#include <tuple>
#include <any>
#include <stdexcept>

#include "simple_trajopt.h"

struct PerchingComputeParams : public BaseComputeParams {
    const double* vars;
    double total_duration;
};

/**
 * @brief Perching-specific trajectory optimization derived from SimpleTrajOpt
 * This class implements the same perching optimization logic as PerchingOptimizer
 * but using the SimpleTrajOpt base class architecture.
 */
class SimplePerching : public SimpleTrajOpt {
   public:
    SimplePerching() : SimpleTrajOpt() {
        // Initialize perching-specific parameters
        landing_speed_offset_ = 1.0;
        tail_length_ = 0.3;
        body_radius_ = 0.1;
        platform_radius_ = 0.5;
        params_.max_body_rate = 3.0;
        params_.max_yaw_rate = 2.0;

        // Initialize state
        has_initial_guess_ = false;
        initial_tail_angle_ = 0.0;
        initial_tail_velocity_params_.setZero();

        target_pos_.setZero();
        target_vel_.setZero();
        landing_att_z_vec_.setZero();
        landing_vel_.setZero();
        landing_basis_x_.setZero();
        landing_basis_y_.setZero();

        setTrajectoryParams(20, 1, 1, 3);

        params_.lbfgs_mem_size = 32;
        params_.lbfgs_past = 3;
        params_.lbfgs_g_epsilon = 0.0;
        params_.lbfgs_min_step = 1e-16;
        params_.lbfgs_delta = 1e-4;
        params_.lbfgs_line_search_type = 0;
    }

    ~SimplePerching() = default;

    // Perching-specific configuration methods (matching PerchingOptimizer API)

    void setRobotParameters(double landing_speed_offset, double tail_length,
                            double body_radius, double platform_radius) {
        assert(landing_speed_offset > 0.0 && "Landing speed offset must be positive");
        assert(tail_length > 0.0 && "Tail length must be positive");
        assert(body_radius > 0.0 && "Body radius must be positive");
        assert(platform_radius > 0.0 && "Platform radius must be positive");

        landing_speed_offset_ = landing_speed_offset;
        tail_length_ = tail_length;
        body_radius_ = body_radius;
        platform_radius_ = platform_radius;
    }

    // Main perching trajectory generation method (matching PerchingOptimizer API)
    bool generateTrajectory(const Eigen::MatrixXd& initial_state,
                            const Eigen::Vector3d& target_pos,
                            const Eigen::Vector3d& target_vel,
                            const Eigen::Quaterniond& landing_quat,
                            int num_pieces,
                            Trajectory& trajectory,
                            const double& replanning_time = -1.0) {
        assert(initial_state.rows() == 3 && initial_state.cols() == 4 && "Initial state must be 3x4 matrix");
        assert(num_pieces > 0 && "Number of pieces must be positive");
        assert(landing_quat.norm() > 0.99 && landing_quat.norm() < 1.01 && "Landing quaternion must be unit quaternion");

        // Store target and landing information
        target_pos_ = target_pos;
        target_vel_ = target_vel;
        quaternionToZAxis(landing_quat, landing_att_z_vec_);

        // Configure parameters for this trajectory
        params_.traj_pieces_num = num_pieces;
        params_.waypoint_num = num_pieces - 1;
        params_.custom_var_dim = 3;  // tail_angle + 2 tail_velocity_params

        // Set initial state from matrix
        DroneState initial_drone_state;
        initial_drone_state.position = initial_state.col(0);
        initial_drone_state.velocity = initial_state.col(1);
        initial_drone_state.acceleration = initial_state.col(2);
        initial_drone_state.jerk = initial_state.col(3);
        setInitialState(initial_drone_state);

        // Store initial state matrix for boundary value computations
        initial_state_matrix_ = initial_state;

        // Compute landing basis vectors
        landing_basis_x_ = landing_att_z_vec_.cross(Eigen::Vector3d(0.0, 0.0, 1.0));
        if (landing_basis_x_.squaredNorm() == 0.0) {
            landing_basis_x_ = landing_att_z_vec_.cross(Eigen::Vector3d(0.0, 1.0, 0.0));
        }
        landing_basis_x_.normalize();
        landing_basis_y_ = landing_att_z_vec_.cross(landing_basis_x_);
        landing_basis_y_.normalize();

        // Compute landing velocity
        landing_vel_ = target_vel_ - landing_att_z_vec_ * landing_speed_offset_;

        // Prepare optimization
        preProcessOptUtils();

        // Initialize optimization variables
        initializeOptimizationVariables(replanning_time);

        // Run L-BFGS optimization
        double min_objective = 0.0;
        int optimization_result = 0;
        iteration_count_ = 0;

        const int total_vars = params_.time_var_dim + 3 * params_.waypoint_num + params_.custom_var_dim;

        optimization_result = lbfgs::lbfgs_optimize(total_vars,
                                                    optimization_vars_,
                                                    &min_objective,
                                                    &SimplePerching::objectiveFunctionWrapper,
                                                    nullptr,
                                                    nullptr,
                                                    this,
                                                    &lbfgs_params_);

        if (optimization_result < 0) {
            return false;
        }

        // Store optimized values
        log_time_var_ = optimization_vars_[0];
        double piece_duration = expC2(log_time_var_);
        optimized_total_duration_ = params_.traj_pieces_num * piece_duration;

        // Extract optimized trajectory
        trajectory = getCurrentTrajectory();
        has_initial_guess_ = true;

        return true;
    }

    bool generateTrajectory(const DroneState& initial_state, Trajectory& trajectory) override {
        // Simplified interface - convert DroneState to matrix format
        Eigen::MatrixXd initial_matrix(3, 4);
        initial_matrix.col(0) = initial_state.position;
        initial_matrix.col(1) = initial_state.velocity;
        initial_matrix.col(2) = initial_state.acceleration;
        initial_matrix.col(3) = initial_state.jerk;

        // Use default target and landing orientation for basic interface
        Eigen::Vector3d default_target(0, 0, 0);
        Eigen::Vector3d default_vel(0, 0, 0);
        Eigen::Quaterniond default_quat(1, 0, 0, 0);

        return generateTrajectory(initial_matrix, default_target, default_vel, default_quat, params_.traj_pieces_num, trajectory);
    }

   protected:
    DroneState computeFinalState(const BaseComputeParams& params) override {
        const auto& perching_params = dynamic_cast<const PerchingComputeParams&>(params);
        const double* vars = perching_params.vars;
        double total_duration = perching_params.total_duration;

        // Extract custom variables
        const double tail_angle = vars[params_.time_var_dim + 3 * params_.waypoint_num];
        const double* tail_velocity_params = vars + params_.time_var_dim + 3 * params_.waypoint_num + 1;

        // Compute tail velocity using the same formula as PerchingOptimizer
        Eigen::Vector3d tail_velocity = landing_vel_ +
                                        tail_velocity_params[0] * landing_basis_x_ +
                                        tail_velocity_params[1] * landing_basis_y_;

        // Compute final state using the same formula as PerchingOptimizer
        DroneState final_state;
        final_state.position = target_pos_ + target_vel_ * total_duration + landing_att_z_vec_ * tail_length_;
        final_state.velocity = tail_velocity;
        final_state.acceleration = forwardThrust(tail_angle) * landing_att_z_vec_ + params_.gravity_vec;
        final_state.jerk.setZero();

        return final_state;
    }

    // Implementation of L-BFGS callback and helpers
    void addTimeIntegralPenalty(double& cost) override {
        Eigen::Vector3d pos, vel, acc, jer;
        Eigen::Vector3d grad_pos_total, grad_vel_total, grad_acc_total, grad_jer_total;
        double cost_temp = 0.0;

        Eigen::Matrix<double, 8, 1> beta0, beta1, beta2, beta3;
        double sigma1 = 0.0;
        double step = minco_optimizer_.t(1) / params_.integration_steps;
        double alpha = 0.0;
        Eigen::Matrix<double, 8, 3> grad_c;
        double grad_t = 0.0;
        double integration_weight = 0.0;

        int inner_loop = params_.integration_steps + 1;

        for (int j = 0; j < inner_loop; ++j) {
            if (j == 0 || j == params_.integration_steps) {
                integration_weight = 1.0 / 6.0;
            } else if (j % 2 == 1) {
                integration_weight = 2.0 / 3.0;
            } else {
                integration_weight = 1.0 / 3.0;
            }
            integration_weight *= step;

            for (int i = 0; i < params_.traj_pieces_num; ++i) {
                alpha = sigma1 + step * j;

                // Calculate beta vectors for polynomial evaluation
                beta0 << 1.0, alpha, alpha * alpha, alpha * alpha * alpha,
                    alpha * alpha * alpha * alpha, alpha * alpha * alpha * alpha * alpha,
                    alpha * alpha * alpha * alpha * alpha * alpha, alpha * alpha * alpha * alpha * alpha * alpha * alpha;
                beta1 << 0.0, 1.0, 2.0 * alpha, 3.0 * alpha * alpha,
                    4.0 * alpha * alpha * alpha, 5.0 * alpha * alpha * alpha * alpha,
                    6.0 * alpha * alpha * alpha * alpha * alpha, 7.0 * alpha * alpha * alpha * alpha * alpha * alpha;
                beta2 << 0.0, 0.0, 2.0, 6.0 * alpha,
                    12.0 * alpha * alpha, 20.0 * alpha * alpha * alpha,
                    30.0 * alpha * alpha * alpha * alpha, 42.0 * alpha * alpha * alpha * alpha * alpha;
                beta3 << 0.0, 0.0, 0.0, 6.0,
                    24.0 * alpha, 60.0 * alpha * alpha,
                    120.0 * alpha * alpha * alpha, 210.0 * alpha * alpha * alpha * alpha;

                // Calculate drone state at current point
                pos = minco_optimizer_.c.block<3, 8>(0, i * 8) * beta0;
                vel = minco_optimizer_.c.block<3, 8>(0, i * 8) * beta1 / minco_optimizer_.t(i + 1);
                acc = minco_optimizer_.c.block<3, 8>(0, i * 8) * beta2 / (minco_optimizer_.t(i + 1) * minco_optimizer_.t(i + 1));
                jer = minco_optimizer_.c.block<3, 8>(0, i * 8) * beta3 / (minco_optimizer_.t(i + 1) * minco_optimizer_.t(i + 1) * minco_optimizer_.t(i + 1));

                // Initialize gradient accumulators
                grad_pos_total.setZero();
                grad_vel_total.setZero();
                grad_acc_total.setZero();
                grad_jer_total.setZero();

                // Standard velocity and acceleration costs
                computeVelocityCost(vel, grad_vel_total, cost_temp);
                cost += cost_temp * integration_weight;

                computeAccelerationCost(acc, grad_acc_total, cost_temp);
                cost += cost_temp * integration_weight;

                // Perching-specific costs
                if (computeThrustCost(acc, grad_acc_total, cost_temp)) {
                    cost += cost_temp * integration_weight;
                }

                if (computeBodyRateCost(acc, jer, grad_acc_total, grad_jer_total, cost_temp)) {
                    cost += cost_temp * integration_weight;
                }

                if (computeFloorCost(pos, grad_pos_total, cost_temp)) {
                    cost += cost_temp * integration_weight;
                }

                // Perching collision cost
                double duration_to_now = (i + alpha) * minco_optimizer_.t(1);
                Eigen::Vector3d car_pos = target_pos_ + target_vel_ * duration_to_now;
                Eigen::Vector3d grad_car_pos = Eigen::Vector3d::Zero();
                if (computePerchingCollisionCost(pos, acc, car_pos, grad_pos_total, grad_acc_total, grad_car_pos, cost_temp)) {
                    cost += cost_temp * integration_weight;
                }

                // Update gradients
                grad_c = Eigen::Matrix<double, 8, 3>::Zero();
                grad_c += beta0 * grad_pos_total.transpose();
                grad_c += (beta1 / minco_optimizer_.t(i + 1)) * grad_vel_total.transpose();
                grad_c += (beta2 / (minco_optimizer_.t(i + 1) * minco_optimizer_.t(i + 1))) * grad_acc_total.transpose();
                grad_c += (beta3 / (minco_optimizer_.t(i + 1) * minco_optimizer_.t(i + 1) * minco_optimizer_.t(i + 1))) * grad_jer_total.transpose();

                minco_optimizer_.gdC.block<3, 8>(0, i * 8) += grad_c.transpose() * integration_weight;

                // Time gradient contribution
                grad_t = -grad_vel_total.dot(vel) / minco_optimizer_.t(i + 1);
                grad_t += -2.0 * grad_acc_total.dot(acc) / minco_optimizer_.t(i + 1);
                grad_t += -3.0 * grad_jer_total.dot(jer) / minco_optimizer_.t(i + 1);
                grad_t += grad_car_pos.dot(target_vel_);

                minco_optimizer_.gdT += grad_t * integration_weight;
                minco_optimizer_.gdT += i * integration_weight * step * grad_car_pos.dot(target_vel_);
            }

            sigma1 += step;
        }
    }

    double objectiveFunction(void* ptr, const double* vars, double* grads, int n) override {
        (void)n;
        iteration_count_++;

        // Clear gradients
        std::fill(grads, grads + n, 0.0);

        // Unpack variables
        const double& log_time_var = vars[0];
        double& grad_log_time = grads[0];

        Eigen::Map<const Eigen::MatrixXd> waypoints(vars + params_.time_var_dim, 3, params_.waypoint_num);
        Eigen::Map<Eigen::MatrixXd> grad_waypoints(grads + params_.time_var_dim, 3, params_.waypoint_num);

        // Custom variables (tail_angle + 2 tail_velocity_params)
        const double tail_angle = vars[params_.time_var_dim + 3 * params_.waypoint_num];
        const double* tail_velocity_params = vars + params_.time_var_dim + 3 * params_.waypoint_num + 1;

        double& grad_tail_angle = grads[params_.time_var_dim + 3 * params_.waypoint_num];
        double* grad_tail_velocity_params = grads + params_.time_var_dim + 3 * params_.waypoint_num + 1;

        // Calculate piece duration
        double piece_duration = expC2(log_time_var);
        double total_duration = params_.traj_pieces_num * piece_duration;

        // Get final state
        PerchingComputeParams compute_params;
        compute_params.vars = vars;
        compute_params.total_duration = total_duration;
        auto final_state = computeFinalState(compute_params);

        // Convert initial and final states to matrix format
        Eigen::MatrixXd initial_matrix(3, 4);
        initial_matrix.col(0) = initial_state_.position;
        initial_matrix.col(1) = initial_state_.velocity;
        initial_matrix.col(2) = initial_state_.acceleration;
        initial_matrix.col(3) = initial_state_.jerk;

        Eigen::MatrixXd final_matrix(3, 4);
        final_matrix.col(0) = final_state.position;
        final_matrix.col(1) = final_state.velocity;
        final_matrix.col(2) = final_state.acceleration;
        final_matrix.col(3) = final_state.jerk;

        // Generate MINCO trajectory
        minco_optimizer_.generate(initial_matrix, final_matrix, waypoints, piece_duration);

        // Start with snap cost
        double cost = minco_optimizer_.getTrajSnapCost();
        minco_optimizer_.calGrads_CT();

        // Add time cost
        minco_optimizer_.gdT += params_.time_weight;
        cost += params_.time_weight * piece_duration;

        // Add time integral penalty (includes perching-specific costs)
        addTimeIntegralPenalty(cost);

        // Calculate MINCO gradients
        minco_optimizer_.calGrads_PT();

        // Set basic gradients
        grad_log_time = minco_optimizer_.gdT * gradTimeTransform(log_time_var);
        grad_waypoints = minco_optimizer_.gdP;

        // Compute custom variable gradients
        computeCustomVariableGradients(vars, grads, cost, minco_optimizer_.gdTail, piece_duration);

        return cost;
    }

   private:
    // Perching-specific member variables (from PerchingOptimizer)
    double landing_speed_offset_;
    double tail_length_;
    double body_radius_;
    double platform_radius_;

    // Landing configuration
    Eigen::Vector3d target_pos_;
    Eigen::Vector3d target_vel_;
    Eigen::Vector3d landing_att_z_vec_;
    Eigen::Vector3d landing_vel_;
    Eigen::Vector3d landing_basis_x_;
    Eigen::Vector3d landing_basis_y_;

    // Initial guess management
    bool has_initial_guess_;
    Eigen::MatrixXd initial_state_matrix_;
    Trajectory initial_trajectory_;
    double initial_tail_angle_;
    Eigen::Vector2d initial_tail_velocity_params_;

    // Helper methods (adapted from PerchingOptimizer)

    static double objectiveFunctionWrapper(void* ptr, const double* vars, double* grads, int n) {
        auto* optimizer = static_cast<SimplePerching*>(ptr);
        return optimizer->objectiveFunction(ptr, vars, grads, n);
    }

    double forwardThrust(double tail_angle) const {
        return params_.thrust_half_range * std::sin(tail_angle) + params_.thrust_half_level;
    }

    static double propagateThrustGradient(double tail_angle, double thrust_gradient, double thrust_half_range) {
        return thrust_half_range * std::cos(tail_angle) * thrust_gradient;
    }

    void initializeOptimizationVariables(double replanning_time = -1.0) {
        double& log_time_var = optimization_vars_[0];
        Eigen::Map<Eigen::MatrixXd> waypoints(optimization_vars_ + params_.time_var_dim, 3, params_.waypoint_num);
        double& tail_angle = optimization_vars_[params_.time_var_dim + 3 * params_.waypoint_num];
        Eigen::Map<Eigen::Vector2d> tail_velocity_params(optimization_vars_ + params_.time_var_dim + 3 * params_.waypoint_num + 1);

        tail_angle = 0.0;
        tail_velocity_params.setZero();

        bool reuse_initial_guess = has_initial_guess_ && replanning_time > 0.0 &&
                                   replanning_time < initial_trajectory_.getTotalDuration();

        if (reuse_initial_guess) {
            // Reuse previous trajectory for warm start
            double remaining_time = initial_trajectory_.getTotalDuration() - replanning_time;
            log_time_var = logC2(remaining_time / params_.traj_pieces_num);

            tail_angle = initial_tail_angle_;
            tail_velocity_params = initial_tail_velocity_params_;

            // Initialize waypoints from previous trajectory
            if (params_.waypoint_num > 0) {
                for (int i = 0; i < params_.waypoint_num; ++i) {
                    double sample_time = replanning_time + (i + 1) * remaining_time / params_.traj_pieces_num;
                    waypoints.col(i) = initial_trajectory_.getPos(sample_time);
                }
            }
        } else {
            // Initialize from boundary value problem
            Eigen::MatrixXd final_boundary(3, 4);
            final_boundary.col(0) = target_pos_;
            final_boundary.col(1) = target_vel_;
            final_boundary.col(2) = forwardThrust(0.0) * landing_att_z_vec_ + params_.gravity_vec;
            final_boundary.col(3).setZero();

            double boundary_duration = (target_pos_ - initial_state_.position).norm() / params_.max_velocity;
            log_time_var = logC2(boundary_duration / params_.traj_pieces_num);

            // Initialize waypoints with straight line
            if (params_.waypoint_num > 0) {
                for (int i = 0; i < params_.waypoint_num; ++i) {
                    double ratio = static_cast<double>(i + 1) / static_cast<double>(params_.traj_pieces_num);
                    waypoints.col(i) = initial_state_.position + ratio * (target_pos_ - initial_state_.position);
                }
            }
        }
    }

    void computeCustomVariableGradients(const double* vars, double* grads, double& cost,
                                        const Eigen::MatrixXd& gdTail, double piece_duration) {
        // Extract custom variables and their gradients
        const double tail_angle = vars[params_.time_var_dim + 3 * params_.waypoint_num];
        const double* tail_velocity_params = vars + params_.time_var_dim + 3 * params_.waypoint_num + 1;

        double& grad_tail_angle = grads[params_.time_var_dim + 3 * params_.waypoint_num];
        double* grad_tail_velocity_params = grads + params_.time_var_dim + 3 * params_.waypoint_num + 1;

        // Add time gradient contribution from final state
        minco_optimizer_.gdT += gdTail.col(0).dot(params_.traj_pieces_num * target_vel_) * piece_duration;

        // Calculate gradient for tail_angle (thrust contribution)
        double thrust_gradient = gdTail.col(2).dot(landing_att_z_vec_);
        grad_tail_angle = propagateThrustGradient(tail_angle, thrust_gradient, params_.thrust_half_range);

        // Calculate gradients for tail_velocity_params
        Eigen::Vector3d grad_tail_velocity = gdTail.col(1);

        grad_tail_velocity_params[0] = grad_tail_velocity.dot(landing_basis_x_);
        grad_tail_velocity_params[1] = grad_tail_velocity.dot(landing_basis_y_);

        // Add tail velocity cost if enabled
        if (params_.terminal_vel_weight > -1.0) {
            Eigen::Vector2d tail_vel_params_vec(tail_velocity_params[0], tail_velocity_params[1]);
            double tail_velocity_norm_sq = tail_vel_params_vec.squaredNorm();
            cost += params_.terminal_vel_weight * tail_velocity_norm_sq;
            grad_tail_velocity_params[0] += params_.terminal_vel_weight * 2.0 * tail_vel_params_vec.x();
            grad_tail_velocity_params[1] += params_.terminal_vel_weight * 2.0 * tail_vel_params_vec.y();
        }
    }

    // Perching-specific cost functions (adapted from PerchingOptimizer)

    bool computePerchingCollisionCost(const Eigen::Vector3d& position, const Eigen::Vector3d& acceleration,
                                      const Eigen::Vector3d& car_position, Eigen::Vector3d& grad_position,
                                      Eigen::Vector3d& grad_acceleration, Eigen::Vector3d& grad_car_position,
                                      double& cost) {
        static double eps = 1e-6;

        double distance_sq = (position - car_position).squaredNorm();
        double safe_radius = platform_radius_ + body_radius_;
        double safe_radius_sq = safe_radius * safe_radius;
        double penalty_distance = safe_radius_sq - distance_sq;
        penalty_distance /= safe_radius_sq;
        double gradient_distance = 0.0;
        double smoothing = smoothedZeroOne(penalty_distance, gradient_distance);
        if (smoothing == 0.0) {
            return false;
        }

        Eigen::Vector3d grad_position_distance = gradient_distance * 2.0 * (car_position - position);
        Eigen::Vector3d grad_car_position_distance = -grad_position_distance;

        Eigen::Vector3d plane_normal = -landing_att_z_vec_;
        double plane_offset = plane_normal.dot(car_position);

        Eigen::Vector3d thrust = acceleration - params_.gravity_vec;
        Eigen::Vector3d body_z_vec = normalizeVector(thrust);

        // Body-to-relative transformation matrix calculation
        Eigen::Matrix<double, 2, 3> matrix_btrt;
        double body_z_x = body_z_vec.x();
        double body_z_y = body_z_vec.y();
        double body_z_z = body_z_vec.z();

        double inv_one_plus_z = 1.0 / (1.0 + body_z_z);

        matrix_btrt(0, 0) = 1.0 - body_z_x * body_z_x * inv_one_plus_z;
        matrix_btrt(0, 1) = -body_z_x * body_z_y * inv_one_plus_z;
        matrix_btrt(0, 2) = -body_z_x;
        matrix_btrt(1, 0) = -body_z_x * body_z_y * inv_one_plus_z;
        matrix_btrt(1, 1) = 1.0 - body_z_y * body_z_y * inv_one_plus_z;
        matrix_btrt(1, 2) = -body_z_y;

        Eigen::Vector2d v2 = matrix_btrt * plane_normal;
        double v2_norm = std::sqrt(v2.squaredNorm() + eps);
        double penalty = plane_normal.dot(position) - (tail_length_ - 0.005) * plane_normal.dot(body_z_vec) -
                         plane_offset + body_radius_ * v2_norm;

        if (penalty > 0.0) {
            double gradient = 0.0;
            cost = smoothedL1(penalty, gradient);

            grad_position = plane_normal;
            grad_car_position = -plane_normal;
            Eigen::Vector2d grad_v2 = body_radius_ * v2 / v2_norm;

            // Compute derivatives of matrix_btrt with respect to body_z_vec components
            Eigen::Matrix<double, 2, 3> dM_dax, dM_day, dM_daz;
            double inv_one_plus_z_sq = inv_one_plus_z * inv_one_plus_z;

            dM_dax(0, 0) = -2.0 * body_z_x * inv_one_plus_z;
            dM_dax(0, 1) = -body_z_y * inv_one_plus_z;
            dM_dax(0, 2) = -1.0;
            dM_dax(1, 0) = -body_z_y * inv_one_plus_z;
            dM_dax(1, 1) = 0.0;
            dM_dax(1, 2) = 0.0;

            dM_day(0, 0) = 0.0;
            dM_day(0, 1) = -body_z_x * inv_one_plus_z;
            dM_day(0, 2) = 0.0;
            dM_day(1, 0) = -body_z_x * inv_one_plus_z;
            dM_day(1, 1) = -2.0 * body_z_y * inv_one_plus_z;
            dM_day(1, 2) = -1.0;

            dM_daz(0, 0) = body_z_x * body_z_x * inv_one_plus_z_sq;
            dM_daz(0, 1) = body_z_x * body_z_y * inv_one_plus_z_sq;
            dM_daz(0, 2) = 0.0;
            dM_daz(1, 0) = body_z_x * body_z_y * inv_one_plus_z_sq;
            dM_daz(1, 1) = body_z_y * body_z_y * inv_one_plus_z_sq;
            dM_daz(1, 2) = 0.0;

            Eigen::Matrix<double, 2, 3> dv2_dzb;
            dv2_dzb.col(0) = dM_dax * plane_normal;
            dv2_dzb.col(1) = dM_day * plane_normal;
            dv2_dzb.col(2) = dM_daz * plane_normal;

            Eigen::Vector3d grad_body_z = dv2_dzb.transpose() * grad_v2 - tail_length_ * plane_normal;

            grad_acceleration = getNormalizationJacobian(thrust).transpose() * grad_body_z;

            gradient *= smoothing;
            grad_position_distance *= cost;
            grad_car_position_distance *= cost;
            cost *= smoothing;
            grad_position = gradient * grad_position + grad_position_distance;
            grad_acceleration *= gradient;
            grad_car_position = gradient * grad_car_position + grad_car_position_distance;

            cost *= params_.collision_weight;
            grad_position *= params_.collision_weight;
            grad_acceleration *= params_.collision_weight;
            grad_car_position *= params_.collision_weight;

            return true;
        }
        return false;
    }
};