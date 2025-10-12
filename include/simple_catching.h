//
// Created by Zhaohong Liu on 25-9-30.
//

#ifndef SIMPLE_CATCHING_H
#define SIMPLE_CATCHING_H

#include "simple_trajopt.h"

struct CatchingComputeParams : public BaseComputeParams {
    const double* vars;
    double total_duration;
};

class SimpleCatching : public SimpleTrajOpt {
   public:
    SimpleCatching() : SimpleTrajOpt() {}
    ~SimpleCatching() = default;

    void setTargetTrajectory(std::shared_ptr<SimpleTrajectory> target_trajectory);

    void setCatchingAttitude(const Eigen::Vector3d& euler_attitude);
    void setCatchingAttitude(const Eigen::Quaterniond& quat_attitude);

    void setInitialGuess(std::shared_ptr<SimpleTrajectory> pursuer_trajectory);

    bool generateTrajectory(const DroneState& initial_state, Trajectory& trajectory) override;

   protected:
    DroneState computeFinalState(const BaseComputeParams& params) override;

    void addTimeIntegralPenalty(double& cost) override;

    double objectiveFunction(void* ptr, const double* vars, double* grads, int n) override;

   private:
    std::shared_ptr<SimpleTrajectory> target_traj_;
};

#endif  // SIMPLE_CATCHING_H
