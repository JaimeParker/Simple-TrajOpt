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

    void setDynamicLimits(double max_vel, double max_acc) {
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

    /**
     * @brief Generates the optimal trajectory. This is the main algorithm template.
     * * This function orchestrates the optimization process by calling virtual "hook"
     * methods that can be overridden by derived classes.
     */
    bool generateTrajectory(const DroneState& initial_state,
                            int num_pieces,
                            Trajectory& trajectory) {
        // TODO: no need for initial guess strategy
    }

   protected:
    // --- VIRTUAL "HOOKS" FOR DERIVED CLASSES ---

    /**
     * @brief Allows derived classes to define problem-specific optimization variables.
     * @param var_count Output parameter for the number of custom variables.
     */
    virtual void defineProblemSpecificVariables(int& var_count) {
        // Base implementation has no extra variables.
        var_count = 0;
    }

    /**
     * @brief Computes the final state (pos, vel, acc, jerk) of the trajectory.
     * @param total_duration The total duration of the trajectory.
     * @param optimization_vars The raw array of current optimization variables.
     * @return A 3x4 matrix representing the final state.
     */
    virtual Eigen::MatrixXd computeFinalState(double total_duration, const double* optimization_vars) = 0;

    /**
     * @brief Adds custom, problem-specific penalty costs during time integration.
     * @param cost The total cost to be accumulated.
     * @param pos Current position.
     * @param acc Current acceleration.
     * @param grad_pos_total Gradient w.r.t. position to be accumulated.
     * @param grad_acc_total Gradient w.r.t. acceleration to be accumulated.
     */
    virtual void addCustomPenalties(double& cost,
                                    const Eigen::Vector3d& pos, const Eigen::Vector3d& vel, const Eigen::Vector3d& acc, const Eigen::Vector3d& jer,
                                    Eigen::Vector3d& grad_pos_total, Eigen::Vector3d& grad_vel_total, Eigen::Vector3d& grad_acc_total, Eigen::Vector3d& grad_jer_total) {
        // Base implementation has no custom penalties.
    }

    // --- CORE (NON-VIRTUAL) IMPLEMENTATION ---

   private:
    // --- L-BFGS CALLBACK AND HELPERS ---

    // (Static helpers like expC2, logC2, smoothedL1, etc. would go here)
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
        // TODO:
    }

    // PRIVATE MEMBERS

    // Trajectory parameters
    int num_pieces_{0};
    int waypoint_dim_{0};
    int integration_steps_{20};
    double time_w_{1.0}, vel_w_{1.0}, acc_w_{1.0}, pos_w_{1.0}, thrust_w_{1.0};
    double collision_w_{1.0}, body_rate_w_{1.0};

    // Dynamic limits
    double vel_max_{10.0}, acc_max_{10.0};
    double thrust_min_{2.0}, thrust_max_{20.0};
    double body_rate_max_{5.0}, body_rate_yaw_max_{5.0};

    // Environment parameters
    Eigen::Vector3d gravity_vec_{0.0, 0.0, -9.8};

    // Drone state
    DroneState initial_state_;

    // MINCO optimizer instance
    minco::MINCO_S4_Uniform minco_optimizer_;
    double* optimization_vars_{nullptr};
};
