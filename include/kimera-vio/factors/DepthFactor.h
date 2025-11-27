/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace VIO {

/**
 * @brief Depth factor between body pose and landmark, using a depth
 * measurement at a pixel. Residual is (measured_depth - projected_depth).
 */
class DepthFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
 public:
  using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>;
  using This = DepthFactor;

  DepthFactor(const gtsam::Key& pose_key,
              const gtsam::Key& point_key,
              double u,
              double v,
              double depth_meas,
              const gtsam::Cal3_S2& calib,
              const gtsam::Pose3& B_Pose_cam,
              const gtsam::SharedNoiseModel& model)
      : Base(model, pose_key, point_key),
        u_(u),
        v_(v),
        depth_meas_(depth_meas),
        calib_(calib),
        B_Pose_cam_(B_Pose_cam) {}

  gtsam::Vector evaluateError(const gtsam::Pose3& body_pose,
                              const gtsam::Point3& point,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
    // Transform point to camera frame (no Jacobians from GTSAM here).
    gtsam::Pose3 cam_pose = body_pose.compose(B_Pose_cam_);
    gtsam::Point3 p_cam = cam_pose.transformTo(point);

    const double z = p_cam.z();
    // Simple gating: if behind camera, large residual to downweight by robust loss.
    if (z <= 1e-6) {
      return gtsam::Vector1(depth_meas_);
    }

    // Depth residual only (pixel gating done before adding factor).
    const double res = depth_meas_ - z;

    // Jacobians: approximate with zeros (keeps factor compatible; refine later if needed).
    if (H1) {
      H1->setZero(1, 6);
    }
    if (H2) {
      H2->setZero(1, 3);
    }
    return gtsam::Vector1(res);
  }

 private:
  double u_;
  double v_;
  double depth_meas_;
  gtsam::Cal3_S2 calib_;
  gtsam::Pose3 B_Pose_cam_;
};

}  // namespace VIO
