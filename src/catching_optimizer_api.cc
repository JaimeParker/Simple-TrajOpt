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
}
