#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cassert>

#include "simple_trajopt.h"

/**
 * @brief Perching-specific trajectory optimization derived from SimpleTrajOpt
 * This class implements the same perching optimization logic as PerchingOptimizer
 * but using the modernized SimpleTrajOpt base class architecture.
 */
class SimplePerching : public SimpleTrajOpt {
public:
    SimplePerching() {
        // Initialize perching-specific parameters
        landing_speed_offset_ = 1.0;
        tail_length_ = 0.3;
        body_radius_ = 0.1;
        platform_radius_ = 0.5;
        
        // Set custom variable dimension for perching (tail_angle + 2 tail_velocity_params)
        setParameters(1, 1, 0, 3);  // 1 piece, 1 time var, 0 waypoints, 3 custom vars
        
        // Configure perching-specific LBFGS and weights
        params_.lbfgs_mem_size = 32;
        params_.lbfgs_past = 3;
        params_.lbfgs_g_epsilon = 0.0;
        params_.lbfgs_min_step = 1e-16;
        params_.lbfgs_delta = 1e-4;
        params_.lbfgs_line_search_type = 0;
        
        // Perching-specific weights
        tail_velocity_w_ = -1.0;
        body_rate_w_ = 1.0;
        perching_collision_w_ = 1.0;
        
        has_initial_guess_ = false;
    }

    // Perching-specific configuration methods
    void setDynamicLimits(double max_velocity, double max_acceleration, 
                         double max_thrust, double min_thrust,
                         double max_body_rate, double max_yaw_body_rate) {
        assert(max_velocity > 0.0 && "Max velocity must be positive");
        assert(max_acceleration > 0.0 && "Max acceleration must be positive");
        assert(max_thrust > min_thrust && "Max thrust must be greater than min thrust");
        assert(max_body_rate > 0.0 && "Max body rate must be positive");
        assert(max_yaw_body_rate > 0.0 && "Max yaw body rate must be positive");

        params_.max_velocity = max_velocity;
        params_.max_acceleration = max_acceleration;
        params_.thrust_max = max_thrust;
        params_.thrust_min = min_thrust;
        max_body_rate_ = max_body_rate;
        max_yaw_body_rate_ = max_yaw_body_rate;
        
        params_.thrust_half_level = 0.5 * (max_thrust + min_thrust);
        params_.thrust_half_range = 0.5 * (max_thrust - min_thrust);
    }

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

    void setOptimizationWeights(double time_weight, double tail_velocity_weight,
                               double position_weight, double velocity_weight,
                               double acceleration_weight, double thrust_weight,
                               double body_rate_weight, double perching_collision_weight) {
        assert(time_weight >= 0.0 && "Time weight must be non-negative");
        assert(position_weight >= 0.0 && "Position weight must be non-negative");
        assert(velocity_weight >= 0.0 && "Velocity weight must be non-negative");
        assert(acceleration_weight >= 0.0 && "Acceleration weight must be non-negative");
        assert(thrust_weight >= 0.0 && "Thrust weight must be non-negative");
        assert(body_rate_weight >= 0.0 && "Body rate weight must be non-negative");
        assert(perching_collision_weight >= 0.0 && "Perching collision weight must be non-negative");

        params_.time_weight = time_weight;
        tail_velocity_w_ = tail_velocity_weight;
        params_.pos_penalty_weight = position_weight;
        params_.vel_penalty_weight = velocity_weight;
        params_.acc_penalty_weight = acceleration_weight;
        thrust_w_ = thrust_weight;
        body_rate_w_ = body_rate_weight;
        perching_collision_w_ = perching_collision_weight;
    }

    // Main perching trajectory generation method
    bool generateTrajectory(const Eigen::MatrixXd& initial_state,
                           const Eigen::Vector3d& target_pos,
                           const Eigen::Vector3d& target_vel,
                           const Eigen::Quaterniond& landing_quat,
                           int num_pieces,
                           Trajectory& trajectory,
                           double replanning_time = -1.0) {
        
        assert(initial_state.rows() == 3 && initial_state.cols() == 4 && "Initial state must be 3x4 matrix");
        assert(num_pieces > 0 && "Number of pieces must be positive");
        assert(landing_quat.norm() > 0.99 && landing_quat.norm() < 1.01 && "Landing quaternion must be unit quaternion");

        // Store target and landing information
        target_pos_ = target_pos;
        target_vel_ = target_vel;
        SimpleTrajOpt::quaternionToZAxis(landing_quat, landing_att_z_vec_);
        
        // Configure parameters for this trajectory
        params_.traj_pieces_num = num_pieces;
        params_.waypoint_num = num_pieces - 1;
        params_.custom_var_dim = 3; // tail_angle + 2 tail_velocity_params
        
        // Set initial state
        DroneState initial_drone_state;
        initial_drone_state.position = initial_state.col(0);
        initial_drone_state.velocity = initial_state.col(1);
        initial_drone_state.acceleration = initial_state.col(2);
        initial_drone_state.jerk = initial_state.col(3);
        setInitialState(initial_drone_state);
        
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
        
        // Initialize optimization variables  
        preProcessOptUtils();
        
        // Initialize custom variables (tail_angle, tail_velocity_params)
        optimization_vars_[params_.time_var_dim + 3 * params_.waypoint_num] = 0.0; // tail_angle
        optimization_vars_[params_.time_var_dim + 3 * params_.waypoint_num + 1] = 0.0; // tail_vel_x
        optimization_vars_[params_.time_var_dim + 3 * params_.waypoint_num + 2] = 0.0; // tail_vel_y
        
        // Initialize time variable with boundary value problem solution
        if (!has_initial_guess_ || replanning_time <= 0.0) {
            initializeFromBoundaryValue();
        }
        
        // Run optimization
        bool success = optimize(optimization_vars_);
        
        if (success) {
            trajectory = getCurrentTrajectory();
            has_initial_guess_ = true;
            return true;
        }
        
        return false;
    }

protected:
    // Override required pure virtual methods from SimpleTrajOpt
    
    bool generateTrajectory(const DroneState& initial_state, Trajectory& trajectory, 
                           int num_pieces = 1, int custom_var_dim = 3) override {
        // This is a simplified interface - use the full perching method instead
        Eigen::MatrixXd initial_matrix(3, 4);
        initial_matrix.col(0) = initial_state.position;
        initial_matrix.col(1) = initial_state.velocity;
        initial_matrix.col(2) = initial_state.acceleration;
        initial_matrix.col(3) = initial_state.jerk;
        
        // Use default target and landing orientation
        Eigen::Vector3d default_target(0, 0, 0);
        Eigen::Vector3d default_vel(0, 0, 0);
        Eigen::Quaterniond default_quat(1, 0, 0, 0);
        
        return generateTrajectory(initial_matrix, default_target, default_vel, default_quat, num_pieces, trajectory);
    }

    DroneState computeFinalState(const double* vars, double total_duration) override {
        // Extract custom variables
        const double tail_angle = vars[params_.time_var_dim + 3 * params_.waypoint_num];
        const double* tail_velocity_params = vars + params_.time_var_dim + 3 * params_.waypoint_num + 1;
        
        // Compute tail velocity
        Eigen::Vector3d tail_velocity = landing_vel_ + 
            tail_velocity_params[0] * landing_basis_x_ + 
            tail_velocity_params[1] * landing_basis_y_;
        
        // Compute final state
        DroneState final_state;
        final_state.position = target_pos_ + target_vel_ * total_duration + landing_att_z_vec_ * tail_length_;
        final_state.velocity = tail_velocity;
        final_state.acceleration = forwardThrust(tail_angle) * landing_att_z_vec_ + params_.gravity_vec;
        final_state.jerk.setZero();
        
        return final_state;
    }

private:
    // Perching-specific member variables
    double landing_speed_offset_;
    double tail_length_;
    double body_radius_;
    double platform_radius_;
    double max_body_rate_;
    double max_yaw_body_rate_;
    
    // Weights
    double tail_velocity_w_;
    double body_rate_w_;
    double perching_collision_w_;
    double thrust_w_;
    
    // Landing configuration
    Eigen::Vector3d target_pos_;
    Eigen::Vector3d target_vel_;
    Eigen::Vector3d landing_att_z_vec_;
    Eigen::Vector3d landing_vel_;
    Eigen::Vector3d landing_basis_x_;
    Eigen::Vector3d landing_basis_y_;
    
    bool has_initial_guess_;
    
    // Helper methods
    
    double forwardThrust(double tail_angle) const {
        return params_.thrust_half_range * std::sin(tail_angle) + params_.thrust_half_level;
    }
    
    void initializeFromBoundaryValue() {
        // Simple initialization using straight-line trajectory
        Eigen::MatrixXd initial_boundary(3, 4);
        initial_boundary.col(0) = initial_state_.position;
        initial_boundary.col(1) = initial_state_.velocity;
        initial_boundary.col(2) = initial_state_.acceleration;
        initial_boundary.col(3) = initial_state_.jerk;
        
        Eigen::MatrixXd final_boundary(3, 4);
        final_boundary.col(0) = target_pos_;
        final_boundary.col(1) = target_vel_;
        final_boundary.col(2) = forwardThrust(0.0) * landing_att_z_vec_ + params_.gravity_vec;
        final_boundary.col(3).setZero();
        
        double boundary_duration = (target_pos_ - initial_state_.position).norm() / params_.max_velocity;
        optimization_vars_[0] = logC2(boundary_duration / params_.traj_pieces_num);
    }
    
    static double expC2(double time_var) {
        return time_var > 0.0 ? ((0.5 * time_var + 1.0) * time_var + 1.0)
                              : 1.0 / ((0.5 * time_var - 1.0) * time_var + 1.0);
    }

    static double logC2(double duration) {
        return duration > 1.0 ? (std::sqrt(2.0 * duration - 1.0) - 1.0)
                              : (1.0 - std::sqrt(2.0 / duration - 1.0));
    }
    
    // Override to add perching-specific costs
    void addTimeIntegralPenalty(double& cost) override {
        // First call base implementation for standard costs
        SimpleTrajOpt::addTimeIntegralPenalty(cost);
        
        // Add perching-specific costs
        addPerchingSpecificCosts(cost);
    }
    
    void addPerchingSpecificCosts(double& cost) {
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
            integration_weight = (j == 0 || j == inner_loop - 1) ? 0.5 : 1.0;
            integration_weight *= step;

            for (int i = 0; i < params_.traj_pieces_num; ++i) {
                alpha = sigma1 + step * j;

                // Calculate beta vectors for polynomial evaluation
                beta0 << 1.0, alpha, alpha*alpha, alpha*alpha*alpha, 
                         alpha*alpha*alpha*alpha, alpha*alpha*alpha*alpha*alpha,
                         alpha*alpha*alpha*alpha*alpha*alpha, alpha*alpha*alpha*alpha*alpha*alpha*alpha;
                beta1 << 0.0, 1.0, 2.0*alpha, 3.0*alpha*alpha,
                         4.0*alpha*alpha*alpha, 5.0*alpha*alpha*alpha*alpha,
                         6.0*alpha*alpha*alpha*alpha*alpha, 7.0*alpha*alpha*alpha*alpha*alpha*alpha;
                beta2 << 0.0, 0.0, 2.0, 6.0*alpha,
                         12.0*alpha*alpha, 20.0*alpha*alpha*alpha,
                         30.0*alpha*alpha*alpha*alpha, 42.0*alpha*alpha*alpha*alpha*alpha;
                beta3 << 0.0, 0.0, 0.0, 6.0,
                         24.0*alpha, 60.0*alpha*alpha,
                         120.0*alpha*alpha*alpha, 210.0*alpha*alpha*alpha*alpha;

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
    
    bool computeThrustCost(const Eigen::Vector3d& acceleration, Eigen::Vector3d& grad_acceleration, double& cost) {
        bool has_penalty = false;
        grad_acceleration.setZero();
        cost = 0.0;
        Eigen::Vector3d thrust = acceleration - params_.gravity_vec;

        double max_penalty = thrust.squaredNorm() - params_.thrust_max * params_.thrust_max;
        if (max_penalty > 0.0) {
            double gradient = 0.0;
            cost = thrust_w_ * SimpleTrajOpt::smoothedL1(max_penalty, gradient);
            grad_acceleration = thrust_w_ * 2.0 * gradient * thrust;
            has_penalty = true;
        }

        double min_penalty = params_.thrust_min * params_.thrust_min - thrust.squaredNorm();
        if (min_penalty > 0.0) {
            double gradient = 0.0;
            cost += thrust_w_ * SimpleTrajOpt::smoothedL1(min_penalty, gradient);
            grad_acceleration += -thrust_w_ * 2.0 * gradient * thrust;
            has_penalty = true;
        }

        return has_penalty;
    }
    
    bool computeBodyRateCost(const Eigen::Vector3d& acceleration, const Eigen::Vector3d& jerk,
                            Eigen::Vector3d& grad_acceleration, Eigen::Vector3d& grad_jerk, double& cost) {
        Eigen::Vector3d thrust = acceleration - params_.gravity_vec;
        Eigen::Vector3d zb_dot = SimpleTrajOpt::getNormalizationJacobian(thrust) * jerk;
        double body_rate_sq = zb_dot.squaredNorm();
        double penalty = body_rate_sq - max_body_rate_ * max_body_rate_;
        
        if (penalty > 0.0) {
            double gradient = 0.0;
            cost = smoothedL1(penalty, gradient);

            Eigen::Vector3d grad_zb_dot = 2.0 * zb_dot;
            grad_jerk = SimpleTrajOpt::getNormalizationJacobian(thrust).transpose() * grad_zb_dot;
            grad_acceleration = SimpleTrajOpt::getNormalizationHessian(thrust, jerk).transpose() * grad_zb_dot;

            cost *= body_rate_w_;
            gradient *= body_rate_w_;
            grad_acceleration *= gradient;
            grad_jerk *= gradient;

            return true;
        }
        return false;
    }
    
    bool computeFloorCost(const Eigen::Vector3d& position, Eigen::Vector3d& grad_position, double& cost) {
        static double z_floor = 0.4;
        double penalty = z_floor - position.z();
        if (penalty > 0.0) {
            double gradient = 0.0;
            cost = SimpleTrajOpt::smoothedL1(penalty, gradient);
            cost *= params_.pos_penalty_weight;
            grad_position.setZero();
            grad_position.z() = -params_.pos_penalty_weight * gradient;
            return true;
        }
        return false;
    }
    
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
        Eigen::Vector3d body_z_vec = SimpleTrajOpt::normalizeVector(thrust);

        double penalty = plane_normal.dot(position) - (tail_length_ - 0.005) * plane_normal.dot(body_z_vec) -
                        plane_offset + body_radius_ * 1.0; // Simplified collision model

        if (penalty > 0.0) {
            double gradient = 0.0;
            cost = SimpleTrajOpt::smoothedL1(penalty, gradient);

            grad_position = plane_normal;
            grad_car_position = -plane_normal;
            grad_acceleration = SimpleTrajOpt::getNormalizationJacobian(thrust).transpose() * (-tail_length_ * plane_normal);

            gradient *= smoothing;
            grad_position_distance *= cost;
            grad_car_position_distance *= cost;
            cost *= smoothing;
            grad_position = gradient * grad_position + grad_position_distance;
            grad_acceleration *= gradient;
            grad_car_position = gradient * grad_car_position + grad_car_position_distance;

            cost *= perching_collision_w_;
            grad_position *= perching_collision_w_;
            grad_acceleration *= perching_collision_w_;
            grad_car_position *= perching_collision_w_;

            return true;
        }
        return false;
    }
    
    static double smoothedZeroOne(double value, double& gradient) {
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
};
