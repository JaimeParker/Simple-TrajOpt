# Simple Trajectory Optimizer

A ROS-independent trajectory optimization library for UAV trajectory planning and perching maneuvers.

## Overview

This project provides trajectory optimization capabilities for UAVs, extracted and refactored from the original ROS-based Fast-Perching implementation. The library includes multiple interfaces for different use cases:

1. **PerchingOptimizer**: Modern C++ interface for UAV perching trajectories with extensive configuration options
2. **TrajOpt**: Original class interface, cleaned of ROS dependencies  
3. **TrajectoryOptimizer**: Alternative redesigned interface (header-only)
4. **SimpleTrajOpt**: Base class for extending with custom trajectory optimization problems

## Features

- **ROS-Independent**: Complete removal of ROS dependencies for use in any C++ project
- **MINCO Trajectory Generation**: Minimum control effort trajectory optimization using polynomial splines
- **L-BFGS Optimization**: Efficient gradient-based optimization with automatic differentiation
- **Collision Avoidance**: Built-in collision detection and avoidance constraints
- **Dynamic Constraints**: Velocity, acceleration, thrust, and body rate limits
- **Perching Optimization**: Specialized for UAV perching maneuvers with landing attitude control
- **Python Bindings**: Full Python API via pybind11 for rapid prototyping and analysis
- **Extensible Architecture**: Base classes for implementing custom trajectory optimization problems

## Dependencies

- **Eigen3**: Linear algebra library (≥3.3)
- **C++14**: Minimum C++ standard required
- **pybind11**: For Python bindings (optional)
- **Python 3.6+**: For Python interface (optional)

## Building

```bash
# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)

# Run examples
./traj_opt_example
./perching_comparison
```

## Usage

### C++ Interface

#### PerchingOptimizer (Recommended)

```cpp
#include "perching_optimizer.h"

// Create optimizer
traj_opt::PerchingOptimizer optimizer;

// Configure dynamic limits
optimizer.setDynamicLimits(
    10.0,  // max velocity (m/s)
    10.0,  // max acceleration (m/s²)
    20.0,  // max thrust (N)
    2.0,   // min thrust (N)
    3.0,   // max body rate (rad/s)
    2.0    // max yaw body rate (rad/s)
);

// Configure robot parameters
optimizer.setRobotParameters(
    1.0,   // landing speed offset (m/s)
    0.3,   // tail length (m)
    0.1,   // body radius (m)
    0.5    // platform radius (m)
);

// Configure optimization weights
optimizer.setOptimizationWeights(
    1.0,   // time weight
    -1.0,  // tail velocity weight (negative = minimize)
    1.0,   // position weight
    1.0,   // velocity weight
    1.0,   // acceleration weight
    1.0,   // thrust weight
    1.0,   // body rate weight
    1.0    // collision weight
);

// Set up initial state [pos, vel, acc, jerk] as 3x4 matrix
Eigen::MatrixXd initial_state(3, 4);
initial_state.col(0) = Eigen::Vector3d(0.0, 0.0, 2.0);   // position
initial_state.col(1) = Eigen::Vector3d(5.0, 0.0, 0.0);   // velocity
initial_state.col(2) = Eigen::Vector3d(0.0, 0.0, -9.81); // acceleration
initial_state.col(3) = Eigen::Vector3d::Zero();           // jerk

// Define target
Eigen::Vector3d target_position(10.0, 0.0, 1.0);
Eigen::Vector3d target_velocity(2.0, 0.0, 0.0);
Eigen::Quaterniond landing_quaternion = Eigen::Quaterniond::Identity();

// Generate trajectory
Trajectory trajectory;
bool success = optimizer.generateTrajectory(
    initial_state, 
    target_position, 
    target_velocity, 
    landing_quaternion, 
    8,  // number of trajectory pieces
    trajectory
);

if (success) {
    std::cout << "Trajectory duration: " << trajectory.getTotalDuration() << "s" << std::endl;
    
    // Evaluate trajectory at any time
    double t = 1.0;
    Eigen::Vector3d pos = trajectory.getPos(t);
    Eigen::Vector3d vel = trajectory.getVel(t);
    Eigen::Vector3d acc = trajectory.getAcc(t);
}
```

#### Original TrajOpt Interface

```cpp
#include "traj_opt.h"

traj_opt::TrajOpt optimizer;
optimizer.setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0);
optimizer.setRobotParameters(1.0, 0.3, 0.1, 0.5);
optimizer.setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

Trajectory result;
bool success = optimizer.generate_traj(
    initial_state, target_pos, target_vel, landing_quat, 
    num_pieces, result
);
```

### Python Interface

```python
# Import the optimizer
from import_utils import import_perching_optimizer
po = import_perching_optimizer()

# Create and configure optimizer
optimizer = (po.PerchingOptimizer()
    .setDynamicLimits(10.0, 10.0, 20.0, 2.0, 3.0, 2.0)
    .setRobotParameters(1.0, 0.3, 0.1, 0.5)
    .setOptimizationWeights(1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    .setIntegrationSteps(20)
    .setDebugMode(False))

# Set up initial state and target
initial_state = np.array([
    [0.0, 5.0, 0.0, 0.0],  # x: pos, vel, acc, jerk
    [0.0, 0.0, 0.0, 0.0],  # y
    [2.0, 0.0, -9.81, 0.0] # z
])

target_pos = np.array([10.0, 0.0, 1.0])
target_vel = np.array([2.0, 0.0, 0.0])
landing_quat = np.array([1.0, 0.0, 0.0, 0.0])

# Generate trajectory
success, trajectory = optimizer.generateTrajectory(
    initial_state, target_pos, target_vel, landing_quat, 8
)

if success:
    print(f"Trajectory duration: {trajectory.getTotalDuration():.2f}s")
    
    # Sample trajectory
    times = np.linspace(0, trajectory.getTotalDuration(), 100)
    positions = np.array([trajectory.getPos(t) for t in times])
    
    # Plot results
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    plt.show()
```

## Architecture

### Core Classes

- **`PerchingOptimizer`**: Main optimization class with full perching capability
- **`TrajOpt`**: Original interface for backward compatibility  
- **`TrajectoryOptimizer`**: Alternative implementation with different API
- **`SimpleTrajOpt`**: Abstract base class for extending with custom problems
- **`Trajectory`**: Polynomial trajectory representation with evaluation methods

### Utilities

- **`MINCO_S4_Uniform`**: Minimum control effort trajectory generation
- **`lbfgs`**: L-BFGS optimization implementation
- **`Piece`**: Individual polynomial trajectory segment
- **Root finding utilities**: For constraint satisfaction

## Code Structure

```
├── include/
│   ├── perching_optimizer.h       # Main PerchingOptimizer class
│   ├── simple_trajopt.h           # Base class for extensions
│   ├── traj_opt.h                 # Original TrajOpt interface
│   ├── traj_opt_perching_v1.h     # TrajectoryOptimizer interface
│   ├── minco.hpp                  # MINCO trajectory generation
│   ├── lbfgs_raw.hpp             # L-BFGS optimization
│   ├── poly_traj_utils.hpp       # Polynomial trajectory utilities
│   └── root_finder.hpp           # Root finding utilities
├── src/
│   ├── traj_opt_perching.cc      # Core implementation
│   └── perching_optimizer_api.cc # Python bindings
├── examples/
│   ├── basic_example.cpp         # Basic C++ usage
│   ├── perching_comparison.cpp   # Compare different optimizers
│   ├── simple_comparison.cpp     # Simple performance test
│   └── precision_test.cpp        # Numerical precision validation
├── scripts/
│   ├── advanced_python_example.py    # Comprehensive Python example
│   ├── test_python_bindings.py       # Python API validation
│   ├── visualize_trajectories.py     # Trajectory visualization
│   └── import_utils.py               # Python import utilities
└── CMakeLists.txt                # Build configuration
```

## Examples

The `examples/` directory contains several demonstration programs:

- **`basic_example`**: Compare PerchingOptimizer vs TrajOpt interfaces
- **`perching_comparison`**: Performance comparison between optimizers
- **`simple_comparison`**: Quick optimization test
- **`precision_test`**: Numerical accuracy validation

The `scripts/` directory contains Python examples:

- **`advanced_python_example.py`**: Full-featured Python demonstration
- **`test_python_bindings.py`**: Python API validation
- **`visualize_trajectories.py`**: Trajectory plotting and analysis

## Extending the Library

### Custom Trajectory Optimization

Extend `SimpleTrajOpt` to create custom optimization problems:

```cpp
#include "simple_trajopt.h"

class MyCustomOptimizer : public SimpleTrajOpt {
public:
    // Implement required virtual functions
    DroneState computeFinalState(const double* vars, double total_duration) override {
        // Define how to compute final state from optimization variables
        // This is called during optimization
    }
    
    bool generateTrajectory(const DroneState& initial_state, Trajectory& trajectory) override {
        // Implement your optimization logic here
        // Set up variables, call optimize(), return result
    }
    
    void addTimeIntegralPenalty(double& cost) override {
        // Add custom cost functions (constraints, penalties, etc.)
    }
};
```

## Performance

Typical optimization performance on modern hardware:
- **Trajectory pieces**: 3-10 segments
- **Optimization time**: 10-100ms per trajectory
- **Integration steps**: 20-50 for good accuracy
- **Convergence**: Usually 20-100 L-BFGS iterations

## Contributing

Contributions welcome! Areas for improvement:
- Additional constraint types
- Alternative optimization algorithms
- More trajectory representations
- Extended Python API features

## Original Attribution

This code is derived from the Fast-Perching project:
- **Repository**: [ZJU-FAST-Lab/Fast-Perching](https://github.com/ZJU-FAST-Lab/Fast-Perching)
- **Original Authors**: ZJU FAST Lab
- **Paper**: "Fast-Perching UAV with Crash-Landing Capability"

## License

Please refer to the original Fast-Perching repository for licensing information.