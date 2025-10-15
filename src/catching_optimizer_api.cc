/**
 * @file catching_optimizer_api.cc
 * @brief Python bindings for CatchingOptimizer using pybind11
 * @author Zhaohong Liu and Claude Sonnet4.5 Agent
 */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "catching_optimizer.h"
#include "simple_trajectory.h"

namespace py = pybind11;

PYBIND11_MODULE(catching_optimizer_py, m) {
    m.doc() = "Python bindings for SimpleCatching trajectory optimization for target interception";

    // First, bind the Trajectory class which is a dependency
    py::class_<Trajectory>(m, "Trajectory")
        .def(py::init<>(), "Default constructor")
        .def(py::init<const std::vector<double>&, const std::vector<CoefficientMat>&>(),
             "Constructor with durations and coefficient matrices",
             py::arg("durations"), py::arg("coefficient_matrices"))

        // Core query methods
        .def("getPieceNum", &Trajectory::getPieceNum,
             "Get the number of trajectory pieces")
        .def("getTotalDuration", &Trajectory::getTotalDuration,
             "Get the total duration of the trajectory")

        // State evaluation methods
        .def("getPos", &Trajectory::getPos,
             "Get position at time t",
             py::arg("t"))
        .def("getVel", &Trajectory::getVel,
             "Get velocity at time t",
             py::arg("t"))
        .def("getAcc", &Trajectory::getAcc,
             "Get acceleration at time t",
             py::arg("t"))

        // Additional useful methods
        .def("getDurations", &Trajectory::getDurations,
             "Get durations of all trajectory pieces")
        .def("getPositions", &Trajectory::getPositions,
             "Get positions at all junctions")
        .def("getMaxVelRate", &Trajectory::getMaxVelRate,
             "Get maximum velocity magnitude")
        .def("getMaxAccRate", &Trajectory::getMaxAccRate,
             "Get maximum acceleration magnitude")
        .def("getMaxThrust", &Trajectory::getMaxThrust,
             "Get maximum thrust magnitude")

        // Junction state methods
        .def("getJuncPos", &Trajectory::getJuncPos,
             "Get position at junction index",
             py::arg("junction_index"))
        .def("getJuncVel", &Trajectory::getJuncVel,
             "Get velocity at junction index",
             py::arg("junction_index"))
        .def("getJuncAcc", &Trajectory::getJuncAcc,
             "Get acceleration at junction index",
             py::arg("junction_index"));

    // Bind SimpleTrajectory abstract base class
    py::class_<SimpleTrajectory, std::shared_ptr<SimpleTrajectory>>(m, "SimpleTrajectory")
        .def("getPosition", &SimpleTrajectory::getPosition,
             "Get target position at time t",
             py::arg("t"))
        .def("getVelocity", &SimpleTrajectory::getVelocity,
             "Get target velocity at time t",
             py::arg("t"))
        .def("getAcceleration", &SimpleTrajectory::getAcceleration,
             "Get target acceleration at time t",
             py::arg("t"))
        .def("getTotalDuration", &SimpleTrajectory::getTotalDuration,
             "Get total duration of the trajectory");

    // Bind DiscreteTrajectory::StateWaypoint nested struct
    py::class_<DiscreteTrajectory::StateWaypoint>(m, "StateWaypoint")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("timestamp", &DiscreteTrajectory::StateWaypoint::timestamp,
                       "Time stamp of this waypoint")
        .def_readwrite("position", &DiscreteTrajectory::StateWaypoint::position,
                       "Position at this waypoint")
        .def_readwrite("velocity", &DiscreteTrajectory::StateWaypoint::velocity,
                       "Velocity at this waypoint")
        .def_readwrite("acceleration", &DiscreteTrajectory::StateWaypoint::acceleration,
                       "Acceleration at this waypoint");

    // Bind DiscreteTrajectory concrete class
    py::class_<DiscreteTrajectory, SimpleTrajectory, std::shared_ptr<DiscreteTrajectory>>(m, "DiscreteTrajectory")
        .def(py::init<>(), "Default constructor")
        .def("addWaypoint", &DiscreteTrajectory::addWaypoint,
             "Add a waypoint to the trajectory",
             py::arg("waypoint"))
        .def("getPosition", &DiscreteTrajectory::getPosition,
             "Get position at time t",
             py::arg("t"))
        .def("getVelocity", &DiscreteTrajectory::getVelocity,
             "Get velocity at time t",
             py::arg("t"))
        .def("getAcceleration", &DiscreteTrajectory::getAcceleration,
             "Get acceleration at time t",
             py::arg("t"))
        .def("getTotalDuration", &DiscreteTrajectory::getTotalDuration,
             "Get total duration of the trajectory");

    // Bind DroneState structure
    py::class_<DroneState>(m, "DroneState")
        .def(py::init<>(), "Default constructor")
        .def(py::init<const Eigen::Vector3d&, const Eigen::Vector3d&, const Eigen::Vector3d&>(),
             "Constructor with position, velocity, and acceleration",
             py::arg("position"), py::arg("velocity"), py::arg("acceleration"))
        .def_readwrite("position", &DroneState::position, "Position vector")
        .def_readwrite("velocity", &DroneState::velocity, "Velocity vector")
        .def_readwrite("acceleration", &DroneState::acceleration, "Acceleration vector")
        .def_readwrite("jerk", &DroneState::jerk, "Jerk vector")
        .def_readwrite("attitude_quat", &DroneState::attitude_quat, "Attitude quaternion [w, x, y, z]")
        .def_readwrite("attitude", &DroneState::attitude, "Attitude euler angles")
        .def_readwrite("body_rate", &DroneState::body_rate, "Body rate vector");

    // Bind the main CatchingOptimizer class
    py::class_<CatchingOptimizer>(m, "CatchingOptimizer")
        .def(py::init<>(), "Default constructor")

        // Configuration methods - designed for method chaining
        .def("setDynamicLimits", [](CatchingOptimizer& self, double max_velocity, double max_acceleration, double max_thrust, double min_thrust, double max_body_rate, double max_yaw_body_rate) -> CatchingOptimizer& {
                 self.setDynamicLimits(max_velocity, max_acceleration, max_thrust, min_thrust, 
                                     max_body_rate, max_yaw_body_rate);
                 return self; }, "Set dynamic constraints for the vehicle", py::arg("max_velocity") = 10.0, py::arg("max_acceleration") = 10.0, py::arg("max_thrust") = 20.0, py::arg("min_thrust") = 2.0, py::arg("max_body_rate") = 3.0, py::arg("max_yaw_body_rate") = 2.0, py::return_value_policy::reference_internal)

        .def("setOptimizationWeights", [](CatchingOptimizer& self, double time_weight, double position_weight, double velocity_weight, double acceleration_weight, double thrust_weight, double body_rate_weight, double terminal_velocity_weight, double collision_weight) -> CatchingOptimizer& {
                 self.setOptimizationWeights(time_weight, position_weight, velocity_weight,
                                           acceleration_weight, thrust_weight, body_rate_weight,
                                           terminal_velocity_weight, collision_weight);
                 return self; }, "Set weights for different cost terms in optimization", py::arg("time_weight") = 1.0, py::arg("position_weight") = 1.0, py::arg("velocity_weight") = 1.0, py::arg("acceleration_weight") = 1.0, py::arg("thrust_weight") = 1.0, py::arg("body_rate_weight") = 1.0, py::arg("terminal_velocity_weight") = -1.0, py::arg("collision_weight") = 1.0, py::return_value_policy::reference_internal)

        .def("setTrajectoryParams", [](CatchingOptimizer& self, int integration_steps, int traj_pieces_num, int time_var_dim, int custom_var_num) -> CatchingOptimizer& {
                 self.setTrajectoryParams(integration_steps, traj_pieces_num, time_var_dim, custom_var_num);
                 return self; }, "Set trajectory generation parameters", py::arg("integration_steps") = 20, py::arg("traj_pieces_num") = 1, py::arg("time_var_dim") = 1, py::arg("custom_var_num") = 3, py::return_value_policy::reference_internal)

        .def("setInitialState", [](CatchingOptimizer& self, const DroneState& initial_state) -> CatchingOptimizer& {
                 self.setInitialState(initial_state);
                 return self; }, "Set initial state of the drone", py::arg("initial_state"), py::return_value_policy::reference_internal)

        .def("setTerminalState", [](CatchingOptimizer& self, const DroneState& terminal_state) -> CatchingOptimizer& {
                 self.setTerminalState(terminal_state);
                 return self; }, "Set desired terminal/final state of the drone", py::arg("terminal_state"), py::return_value_policy::reference_internal)

        .def("setTargetTrajectory", [](CatchingOptimizer& self, std::shared_ptr<SimpleTrajectory> target_trajectory) -> CatchingOptimizer& {
                 self.setTargetTrajectory(target_trajectory);
                 return self; }, "Set the target trajectory to intercept", py::arg("target_trajectory"), py::return_value_policy::reference_internal)

        .def("setCatchingAttitude", py::overload_cast<const Eigen::Vector3d&>(&CatchingOptimizer::setCatchingAttitude), "Set desired catching attitude using euler angles (roll, pitch, yaw)", py::arg("euler_attitude"))

        .def("setCatchingAttitude", py::overload_cast<const Eigen::Quaterniond&>(&CatchingOptimizer::setCatchingAttitude), "Set desired catching attitude using quaternion", py::arg("quat_attitude"))

        // Main trajectory generation method - wrapped to return tuple
        .def("generateTrajectory", [](CatchingOptimizer& self) -> py::tuple {
                 Trajectory trajectory;
                 bool success = self.generateTrajectory(trajectory);
                 return py::make_tuple(success, trajectory); }, "Generate catching/interception trajectory. Returns (success, trajectory) tuple")

        // Getter methods
        .def("getIterationCount", &CatchingOptimizer::getIterationCount, "Get the number of optimization iterations performed")
        .def("getCurrentTrajectory", &CatchingOptimizer::getCurrentTrajectory, "Get the current trajectory from MINCO optimizer");

    // Add module-level utility functions
    m.attr("__version__") = "1.0.0";

    // Helper functions for creating common matrix types
    m.def("createInitialState", [](const Eigen::Vector3d& pos, const Eigen::Vector3d& vel, const Eigen::Vector3d& acc, const Eigen::Vector3d& jerk) -> Eigen::MatrixXd {
              Eigen::MatrixXd state(3, 4);
              state.col(0) = pos;
              state.col(1) = vel;
              state.col(2) = acc;
              state.col(3) = jerk;
              return state; }, "Helper function to create initial state matrix", py::arg("position"), py::arg("velocity"), py::arg("acceleration") = Eigen::Vector3d::Zero(), py::arg("jerk") = Eigen::Vector3d::Zero());

    m.def("createQuaternion", [](double w, double x, double y, double z) { return Eigen::Quaterniond(w, x, y, z).normalized(); }, "Helper function to create and normalize a quaternion", py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"));

    m.def("createDroneState", [](const Eigen::Vector3d& pos, const Eigen::Vector3d& vel, const Eigen::Vector3d& acc) -> DroneState { return DroneState(pos, vel, acc); }, "Helper function to create a DroneState", py::arg("position"), py::arg("velocity"), py::arg("acceleration") = Eigen::Vector3d::Zero());

    // Bind Eigen::Quaterniond explicitly
    py::class_<Eigen::Quaterniond>(m, "Quaterniond")
        .def(py::init<double, double, double, double>(),
             "Constructor with w, x, y, z components",
             py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def("w", [](const Eigen::Quaterniond& q) { return q.w(); }, "Get w component")
        .def("x", [](const Eigen::Quaterniond& q) { return q.x(); }, "Get x component")
        .def("y", [](const Eigen::Quaterniond& q) { return q.y(); }, "Get y component")
        .def("z", [](const Eigen::Quaterniond& q) { return q.z(); }, "Get z component")
        .def("normalized", [](const Eigen::Quaterniond& q) { return q.normalized(); }, "Get normalized quaternion")
        .def("norm", [](const Eigen::Quaterniond& q) { return q.norm(); }, "Get quaternion norm");

    // Utility conversion functions
    m.def("euler2Quaternion", &CatchingOptimizer::euler2Quaternion,
          "Convert euler angles (roll, pitch, yaw) to quaternion",
          py::arg("euler"));

    m.def("q2EulerAngle", &CatchingOptimizer::q2EulerAngle,
          "Convert quaternion to euler angles (roll, pitch, yaw)",
          py::arg("quaternion"), py::arg("euler"));

    m.def("rotB2A", &CatchingOptimizer::rotB2A,
          "Get rotation matrix from body frame to inertial frame",
          py::arg("attitude"));

    m.def("rotBody2World", &CatchingOptimizer::rotB2ody2World,
          "Get rotation matrix from body frame to world frame",
          py::arg("attitude"));
}
