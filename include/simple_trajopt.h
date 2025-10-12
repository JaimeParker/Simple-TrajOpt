#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <memory>
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

struct BaseComputeParams {
    virtual ~BaseComputeParams() = default;
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
        // optimization_vars_ is managed as raw pointer due to L-BFGS API requirements
        if (optimization_vars_ != nullptr) {
            delete[] optimization_vars_;
            optimization_vars_ = nullptr;
        }
    }

    // --- PUBLIC API ---
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
    }

    void setInitialState(const DroneState& initial_state) { initial_state_ = initial_state; }
    int getIterationCount() const { return iteration_count_; }
    Trajectory getCurrentTrajectory() const { return minco_optimizer_.getTraj(); }

    virtual bool generateTrajectory(const DroneState& initial_state, Trajectory& trajectory) = 0;

    static Eigen::Quaterniond euler2Quaternion(const Eigen::Vector3d &euler) {
        return Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX());
    }

    static void q2EulerAngle(const Eigen::Quaterniond &q, Eigen::Vector3d &euler) {
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

    static Eigen::Matrix3d rotB2A(const Eigen::Vector3d &att) {
        double phi = att[0];
        double theta = att[1];

        Eigen::Matrix3d rotation_matrix;
        rotation_matrix << 1, tan(theta) * sin(phi), tan(theta) * cos(phi),
                        0, cos(phi), -sin(phi),
                        0, sin(phi) / (cos(theta) + 1e-8), cos(phi) / (cos(theta) + 1e-8);

        return rotation_matrix;
    }

    static Eigen::Matrix3d rotB2ody2World(const Eigen::Vector3d &att) {
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

   protected:
    // --- PURE VIRTUAL ---
    // virtual DroneState computeFinalState(const double* vars, double total_duration) = 0;
    virtual DroneState computeFinalState(const BaseComputeParams& params) = 0;

    // --- VIRTUAL "HOOKS" FOR DERIVED CLASSES ---
    virtual DroneState generateTerminalState(const Eigen::Vector3d& terminal_pos,
                                             const Eigen::Vector3d& terminal_vel,
                                             const Eigen::Vector3d& terminal_acc) {
        DroneState terminal_state;
        terminal_state.position = terminal_pos;
        terminal_state.velocity = terminal_vel;
        terminal_state.acceleration = terminal_acc;

        // Default values for jerk, attitude, and body rate
        return terminal_state;
    }

    virtual DroneState generateTerminalState(const Trajectory& trajectory, double time_point) {
        Eigen::Vector3d terminal_pos = trajectory.getPos(time_point);
        Eigen::Vector3d terminal_vel = trajectory.getVel(time_point);
        Eigen::Vector3d terminal_acc = trajectory.getAcc(time_point);
        return generateTerminalState(terminal_pos, terminal_vel, terminal_acc);
    }

    virtual int earlyExitCallback(const double* vars,
                                  const double* grads,
                                  const double fx,
                                  const double xnorm,
                                  const double gnorm,
                                  const double step,
                                  int n,
                                  int k,
                                  int ls) {
        return 0;
    }

    virtual void preProcessOptUtils() {
        // This function should be called by the user manually after setting parameters and before optimization
        lbfgs_params_.mem_size = params_.lbfgs_mem_size;
        lbfgs_params_.past = params_.lbfgs_past;
        lbfgs_params_.g_epsilon = params_.lbfgs_g_epsilon;
        lbfgs_params_.min_step = params_.lbfgs_min_step;
        lbfgs_params_.delta = params_.lbfgs_delta;
        lbfgs_params_.line_search_type = params_.lbfgs_line_search_type;

        params_.traj_pieces_num = std::max(1, params_.traj_pieces_num);
        params_.waypoint_num = std::max(0, params_.traj_pieces_num - 1);
        params_.time_var_dim = std::max(1, params_.time_var_dim);

        const int total_vars = params_.time_var_dim + 3 * params_.waypoint_num + params_.custom_var_dim;
        if (optimization_vars_ != nullptr) {
            delete[] optimization_vars_;
        }
        optimization_vars_ = new double[total_vars];

        params_.thrust_half_level = 0.5 * (params_.max_thrust + params_.min_thrust);
        params_.thrust_half_range = 0.5 * (params_.max_thrust - params_.min_thrust);

        minco_optimizer_.reset(params_.traj_pieces_num);
    }

    virtual Eigen::Map<Eigen::MatrixXd>
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

    // --- L-BFGS CALLBACK AND HELPERS ---
    virtual void addTimeIntegralPenalty(double& cost) = 0;
    virtual double objectiveFunction(void* ptr,
                                     const double* vars,
                                     double* grads,
                                     int n) = 0;

    virtual void computeVelocityCost(const Eigen::Vector3d& velocity,
                                     Eigen::Vector3d& grad_velocity,
                                     double& cost_velocity) {
        double vel_penalty = velocity.squaredNorm() - params_.max_velocity * params_.max_velocity;
        if (vel_penalty > 0.0) {
            double smoothed_grad = 0.0;
            cost_velocity = smoothedL1(vel_penalty, smoothed_grad) * params_.vel_penalty_weight;
            grad_velocity = 2.0 * velocity * smoothed_grad * params_.vel_penalty_weight;
        } else {
            cost_velocity = 0.0;
            grad_velocity.setZero();
        }
    }

    virtual void computeAccelerationCost(const Eigen::Vector3d& acceleration,
                                         Eigen::Vector3d& grad_acceleration,
                                         double& cost_acceleration) {
        double acc_penalty = acceleration.squaredNorm() - params_.max_acceleration * params_.max_acceleration;
        if (acc_penalty > 0.0) {
            double smoothed_grad = 0.0;
            cost_acceleration = smoothedL1(acc_penalty, smoothed_grad) * params_.acc_penalty_weight;
            grad_acceleration = 2.0 * acceleration * smoothed_grad * params_.acc_penalty_weight;
        } else {
            cost_acceleration = 0.0;
            grad_acceleration.setZero();
        }
    }

    virtual bool computeFloorCost(const Eigen::Vector3d& position,
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

    virtual bool computeBodyRateCost(const Eigen::Vector3d& acceleration,
                                     const Eigen::Vector3d& jerk,
                                     Eigen::Vector3d& grad_acceleration,
                                     Eigen::Vector3d& grad_jerk, double& cost) {
        Eigen::Vector3d thrust = acceleration - params_.gravity_vec;
        Eigen::Vector3d zb_dot = getNormalizationJacobian(thrust) * jerk;
        double body_rate_sq = zb_dot.squaredNorm();
        auto penalty = body_rate_sq - params_.max_body_rate * params_.max_body_rate;

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

    virtual bool computeThrustCost(const Eigen::Vector3d& acceleration,
                                   Eigen::Vector3d& grad_acceleration, double& cost) {
        bool has_penalty = false;
        grad_acceleration.setZero();
        cost = 0.0;
        Eigen::Vector3d thrust = acceleration - params_.gravity_vec;

        double max_penalty = thrust.squaredNorm() - params_.max_thrust * params_.max_thrust;
        if (max_penalty > 0.0) {
            double gradient = 0.0;
            cost = params_.thrust_weight * smoothedL1(max_penalty, gradient);
            grad_acceleration = params_.thrust_weight * 2.0 * gradient * thrust;
            has_penalty = true;
        }

        double min_penalty = params_.min_thrust * params_.min_thrust - thrust.squaredNorm();
        if (min_penalty > 0.0) {
            double gradient = 0.0;
            cost += params_.thrust_weight * smoothedL1(min_penalty, gradient);
            grad_acceleration += -params_.thrust_weight * 2.0 * gradient * thrust;
            has_penalty = true;
        }

        return has_penalty;
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

    static double smoothedL1(double value,
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

    static double smoothedZeroOne(double value,
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

    static const Trajectory& getDefaultTrajectory() {
        static Trajectory default_trajectory;
        return default_trajectory;
    }

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

    // PROTECTED MEMBERS - accessible by derived classes
    DroneState initial_state_;

    // optimizer
    TrajOptParameters params_;
    minco::MINCO_S4_Uniform minco_optimizer_;
    lbfgs::lbfgs_parameter_t lbfgs_params_;

    double* optimization_vars_ = nullptr;
    int iteration_count_ = 0;
    double log_time_var_ = 0.0;
    double optimized_total_duration_ = 0.0;
};


class SimpleTrajectory {
public:
    // Virtual destructor for proper inheritance
    virtual ~SimpleTrajectory() = default;

    // Pure virtual functions to get the target's state at any time 't'
    // TODO: realize these functions in base class, and allow derived classes to override if needed
    virtual Eigen::Vector3d getPosition(double t);
    virtual Eigen::Vector3d getVelocity(double t);
    virtual Eigen::Vector3d getAcceleration(double t);
};