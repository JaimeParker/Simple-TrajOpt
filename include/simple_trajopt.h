#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <vector>

#include "lbfgs_raw.hpp"
#include "minco.hpp"
#include "poly_traj_utils.hpp"

struct DroneState {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Vector3d acceleration;
    Eigen::Vector3d jerk;
    Eigen::Vector4d attitude_quat;  // [w, x, y, z]
    Eigen::Vector3d attitude;
    Eigen::Vector3d body_rate;

    DroneState() {
        position.setZero();
        velocity.setZero();
        acceleration.setZero();
        jerk.setZero();
        attitude.setZero();
        body_rate.setZero();
        attitude_quat << 1.0, 0.0, 0.0, 0.0;
    }

    DroneState(const Eigen::Vector3d& pos, const Eigen::Vector3d& vel, const Eigen::Vector3d& acc) {
        position = pos;
        velocity = vel;
        acceleration = acc;
        jerk.setZero();
        attitude.setZero();
        body_rate.setZero();
        attitude_quat << 1.0, 0.0, 0.0, 0.0;
    }
};

/**
 * @brief A base class for simplified trajectory optimization.
 * * This class provides the core machinery for MINCO-based trajectory optimization using L-BFGS.
 * It is designed to be extended by derived classes that implement problem-specific logic,
 * such as custom cost functions, final state definitions, and initial guess strategies.
 * Still in development.
 */
class SimpleTrajOpt {
   public:
    // Constructor and virtual destructor for proper inheritance
    SimpleTrajOpt() = default;
    virtual ~SimpleTrajOpt() {
        if (optimization_vars_ != nullptr) {
            delete[] optimization_vars_;
        }
    }

    // --- PUBLIC API ---

    void setDynamicLimits(double max_velocity,
                          double max_acceleration,
                          double max_thrust,
                          double min_thrust,
                          double max_body_rate,
                          double max_yaw_body_rate) {
        vel_max_ = max_velocity;
        acc_max_ = max_acceleration;
        thrust_max_ = max_thrust;
        thrust_min_ = min_thrust;
        body_rate_max_ = max_body_rate;
        yaw_rate_max_ = max_yaw_body_rate;
    }

    void setWeights(double time_w, double vel_w, double acc_w) {
        // TODO: add more weights
        time_w_ = time_w;
        vel_w_ = vel_w;
        acc_w_ = acc_w;
    }

    void setIntegrationSteps(int steps) {
        integration_steps_ = steps;
    }

   protected:
    // --- VIRTUAL "HOOKS" FOR DERIVED CLASSES ---

    virtual DroneState computeFinalState(double total_duration, const double* optimization_vars) = 0;

    // template <typename... Args>
    // DroneState computeFinalState(Args&&... args) {
    //     // This is a template and cannot be virtual.
    //     // The actual implementation will be in the derived class.
    //     // This base implementation is a placeholder.
    //     static_assert(sizeof...(Args) < 0, "computeFinalState must be implemented in the derived class.");
    //     return DroneState{};
    // }

    virtual bool generateTrajectory(const DroneState& initial_state,
                                    Trajectory& trajectory,
                                    int num_pieces = 1,
                                    int custom_var_dim = 3) = 0;

    virtual void preProcessOptUtils(int num_pieces,
                                    int custom_var_dim = 3,
                                    int time_var_dim = 1) {
        traj_pieces_num_ = num_pieces;
        waypoint_num_ = traj_pieces_num_ - 1;
        time_var_dim_ = time_var_dim;

        const int total_vars = time_var_dim_ + 3 * waypoint_num_ + custom_var_dim;
        if (optimization_vars_ != nullptr) {
            delete[] optimization_vars_;
        }
        optimization_vars_ = new double[total_vars];

        thrust_half_level_ = 0.5 * (thrust_max_ + thrust_min_);
        thrust_half_range_ = 0.5 * (thrust_max_ - thrust_min_);

        minco_optimizer_.reset(traj_pieces_num_);

        // TODO: tail_angle, tail_velocity_params, and land_vel_ are not handled yet
        // tail_angle is used to set terminal acceleration
    }

    virtual Eigen::Map<Eigen::MatrixXd>
    getIntermediateWaypoints(const Eigen::Vector3d& start_pos, const Eigen::Vector3d& end_pos,
                             int num_pieces, bool use_straight_line = true,
                             Trajectory& trajectory = getDefaultTrajectory()) {
        Eigen::Map<Eigen::MatrixXd> waypoints(optimization_vars_ + time_var_dim_, 3, waypoint_num_);

        if (!use_straight_line && trajectory.getPieceNum() > 0) {
            double total_duration = trajectory.getTotalDuration();
            for (int i = 0; i < waypoint_num_; ++i) {
                double ratio = static_cast<double>(i + 1) / static_cast<double>(num_pieces);
                double sample_time = ratio * total_duration;
                waypoints.col(i) = trajectory.getPos(sample_time);
            }
        } else {
            for (int i = 0; i < waypoint_num_; ++i) {
                double ratio = static_cast<double>(i + 1) / static_cast<double>(num_pieces);
                waypoints.col(i) = start_pos + ratio * (end_pos - start_pos);
            }
        }

        return waypoints;
    }

    // --- CORE (NON-VIRTUAL) IMPLEMENTATION ---

   private:
    // --- L-BFGS CALLBACK AND HELPERS ---

    static double objectiveFunction(void* ptr,
                                    const double* vars,
                                    double* grads,
                                    int n) {
        (void)n;
        auto* optimizer = static_cast<SimpleTrajOpt*>(ptr);

        // Unpack variables
        const double& log_time_var = vars[0];
        double& grad_log_time = grads[0];

        Eigen::Map<const Eigen::MatrixXd> waypoints(vars + 1, 3, optimizer->waypoint_num_);
        Eigen::Map<Eigen::MatrixXd> grad_waypoints(grads + 1, 3, optimizer->waypoint_num_);

        // Calculate piece duration
        double piece_duration = expC2(log_time_var);
        double total_duration = optimizer->traj_pieces_num_ * piece_duration;

        // Get final state from derived class
        auto final_state = optimizer->computeFinalState(total_duration, vars);

        // Convert initial state to matrix format
        Eigen::MatrixXd initial_matrix(3, 4);
        initial_matrix.col(0) = optimizer->initial_state_.position;
        initial_matrix.col(1) = optimizer->initial_state_.velocity;
        initial_matrix.col(2) = optimizer->initial_state_.acceleration;
        initial_matrix.col(3) = optimizer->initial_state_.jerk;
        Eigen::MatrixXd final_matrix(3, 4);
        final_matrix.col(0) = final_state.position;
        final_matrix.col(1) = final_state.velocity;
        final_matrix.col(2) = final_state.acceleration;
        final_matrix.col(3) = final_state.jerk;

        // Generate trajectory for current iteration
        optimizer->minco_optimizer_.generate(initial_matrix, final_matrix, waypoints, piece_duration);

        // Initialize cost with snap cost
        double cost = optimizer->minco_optimizer_.getTrajSnapCost();
        optimizer->minco_optimizer_.calGrads_CT();

        // Add time integral penalty
        optimizer->addTimeIntegralPenalty(cost);

        // Propagate gradients
        optimizer->minco_optimizer_.calGrads_PT();

        // Add time cost
        optimizer->minco_optimizer_.gdT += optimizer->time_w_;
        cost += optimizer->time_w_ * piece_duration;

        // Calculate final gradients
        grad_log_time = optimizer->minco_optimizer_.gdT * gradTimeTransform(log_time_var);
        grad_waypoints = optimizer->minco_optimizer_.gdP;

        return cost;
    }

    void addTimeIntegralPenalty(double& cost) {
        Eigen::Vector3d pos, vel, acc, jer, snap;
        Eigen::Vector3d grad_pos_total, grad_vel_total, grad_acc_total, grad_jer_total;
        double cost_temp = 0.0;

        Eigen::Matrix<double, 8, 1> beta0, beta1, beta2, beta3, beta4;
        double sigma1 = 0.0;
        double step = 0.0;
        double alpha = 0.0;
        Eigen::Matrix<double, 8, 3> grad_c;
        double grad_t = 0.0;
        double integration_weight = 0.0;

        int inner_loop = integration_steps_ + 1;
        step = minco_optimizer_.t(1) / integration_steps_;

        for (int j = 0; j < inner_loop; ++j) {
            if (j == 0 || j == integration_steps_) {
                integration_weight = 1.0 / 6.0;
            } else if (j % 2 == 1) {
                integration_weight = 2.0 / 3.0;
            } else {
                integration_weight = 1.0 / 3.0;
            }
            integration_weight *= step;

            for (int i = 0; i < traj_pieces_num_; ++i) {
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
                beta4 << 0.0, 0.0, 0.0, 0.0,
                    24.0, 120.0 * alpha,
                    360.0 * alpha * alpha, 840.0 * alpha * alpha * alpha;

                // Calculate drone state at current point
                pos = minco_optimizer_.c.block<3, 8>(0, i * 8) * beta0;
                vel = minco_optimizer_.c.block<3, 8>(0, i * 8) * beta1 / minco_optimizer_.t(i + 1);
                acc = minco_optimizer_.c.block<3, 8>(0, i * 8) * beta2 / (minco_optimizer_.t(i + 1) * minco_optimizer_.t(i + 1));
                jer = minco_optimizer_.c.block<3, 8>(0, i * 8) * beta3 / (minco_optimizer_.t(i + 1) * minco_optimizer_.t(i + 1) * minco_optimizer_.t(i + 1));
                snap = minco_optimizer_.c.block<3, 8>(0, i * 8) * beta4 / (minco_optimizer_.t(i + 1) * minco_optimizer_.t(i + 1) * minco_optimizer_.t(i + 1) * minco_optimizer_.t(i + 1));

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

                // TODO: to allow user-defined costs here, use a function pointer/reference in params

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

                minco_optimizer_.gdT += grad_t * integration_weight;
            }

            sigma1 += step;
        }
    }

    void computeVelocityCost(const Eigen::Vector3d& velocity,
                             Eigen::Vector3d& grad_velocity,
                             double& cost_velocity) {
        double vel_penalty = velocity.squaredNorm() - vel_max_ * vel_max_;
        if (vel_penalty > 0.0) {
            double smoothed_grad = 0.0;
            cost_velocity = smoothedL1(vel_penalty, smoothed_grad) * vel_w_;
            grad_velocity = 2.0 * velocity * smoothed_grad * vel_w_;
        } else {
            cost_velocity = 0.0;
            grad_velocity.setZero();
        }
    }

    void computeAccelerationCost(const Eigen::Vector3d& acceleration,
                                 Eigen::Vector3d& grad_acceleration,
                                 double& cost_acceleration) {
        double acc_penalty = acceleration.squaredNorm() - acc_max_ * acc_max_;
        if (acc_penalty > 0.0) {
            double smoothed_grad = 0.0;
            cost_acceleration = smoothedL1(acc_penalty, smoothed_grad) * acc_w_;
            grad_acceleration = 2.0 * acceleration * smoothed_grad * acc_w_;
        } else {
            cost_acceleration = 0.0;
            grad_acceleration.setZero();
        }
    }

    // STATIC UTILITY FUNCTIONS
    static bool quaternionToZAxis(const Eigen::Quaterniond& quat,
                                  Eigen::Vector3d& z_axis) {
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

    static double smoothedL1(const double& value,
                             double& gradient) {
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

    static double smoothedZeroOne(const double& value,
                                  double& gradient) {
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

    static Trajectory& getDefaultTrajectory() {
        static Trajectory default_trajectory;
        return default_trajectory;
    }

    static void solveBoundaryValueProblem(const double& duration,
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

    // PRIVATE MEMBERS

    // Trajectory parameters
    int traj_pieces_num_{0};
    int waypoint_num_{0};
    int integration_steps_{20};
    double time_w_{1.0}, vel_w_{1.0}, acc_w_{1.0}, pos_w_{1.0}, thrust_w_{1.0};
    double collision_w_{1.0}, body_rate_w_{1.0};
    int time_var_dim_ = 1;

    // Dynamic limits
    double vel_max_{10.0}, acc_max_{10.0};
    double thrust_min_{2.0}, thrust_max_{20.0};
    double body_rate_max_{5.0}, yaw_rate_max_{5.0};
    double thrust_half_level_{10.0}, thrust_half_range_{8.0};

    // Environment parameters
    Eigen::Vector3d gravity_vec_{0.0, 0.0, -9.8};

    // Drone state
    DroneState initial_state_;

    // MINCO optimizer instance
    minco::MINCO_S4_Uniform minco_optimizer_;
    double* optimization_vars_{nullptr};
};
