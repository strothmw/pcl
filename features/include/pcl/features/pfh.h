/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_PFH_H_
#define PCL_PFH_H_

#include <pcl/point_types.h>
#include <pcl/features/feature.h>
#include <pcl/features/pfh_tools.h>
#include <map>
#include <unordered_map>

#include <cmath>
#include <assert.h>

namespace pcl
{
  /** \brief PFHEstimation estimates the Point Feature Histogram (PFH) descriptor for a given point cloud dataset
    * containing points and normals.
    *
    * A commonly used type for PointOutT is pcl::PFHSignature125.
    *
    * \note If you use this code in any academic work, please cite:
    *
    *   - R.B. Rusu, N. Blodow, Z.C. Marton, M. Beetz.
    *     Aligning Point Cloud Views using Persistent Feature Histograms.
    *     In Proceedings of the 21st IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),
    *     Nice, France, September 22-26 2008.
    *   - R.B. Rusu, Z.C. Marton, N. Blodow, M. Beetz.
    *     Learning Informative Point Classes for the Acquisition of Object Model Maps.
    *     In Proceedings of the 10th International Conference on Control, Automation, Robotics and Vision (ICARCV),
    *     Hanoi, Vietnam, December 17-20 2008.
    *
    * \attention 
    * The convention for PFH features is:
    *   - if a query point's nearest neighbors cannot be estimated, the PFH feature will be set to NaN 
    *     (not a number)
    *   - it is impossible to estimate a PFH descriptor for a point that
    *     doesn't have finite 3D coordinates. Therefore, any point that contains
    *     NaN data on x, y, or z, will have its PFH feature property set to NaN.
    *
    * \note The code is stateful as we do not expect this class to be multicore parallelized. Please look at
    * \ref FPFHEstimationOMP for examples on parallel implementations of the FPFH (Fast Point Feature Histogram).
    *
    * \author Radu B. Rusu
    * \ingroup features
    */
  template <typename PointInT, typename PointNT, typename PointOutT = pcl::PFHSignature125, bool T_optimize_for_organized = false>
  class PFHEstimation : public FeatureFromNormals<PointInT, PointNT, PointOutT>, public PFHPairFeaturesManagedCache<PointInT, PointNT>
  {
    public:
      typedef boost::shared_ptr<PFHEstimation<PointInT, PointNT, PointOutT> > Ptr;
      typedef boost::shared_ptr<const PFHEstimation<PointInT, PointNT, PointOutT> > ConstPtr;
      using Feature<PointInT, PointOutT>::feature_name_;
      using Feature<PointInT, PointOutT>::getClassName;
      using Feature<PointInT, PointOutT>::indices_;
      using Feature<PointInT, PointOutT>::k_;
      using Feature<PointInT, PointOutT>::search_parameter_;
      using Feature<PointInT, PointOutT>::surface_;
      using Feature<PointInT, PointOutT>::input_;
      using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;

      typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
      typedef typename Feature<PointInT, PointOutT>::PointCloudIn  PointCloudIn;

      /** \brief Empty constructor. 
        * Sets \a use_cache_ to false, \a nr_subdiv_ to 5, and the internal maximum cache size to 1GB.
        */
      PFHEstimation () : 
        nr_subdiv_ (5), 
        pfh_histogram_ (),
        pfh_tuple_ (),
        d_pi_ (1.0f / (2.0f * static_cast<float> (M_PI)))
      {
        feature_name_ = "PFHEstimation";
      };

      /** \brief Estimate the PFH (Point Feature Histograms) individual signatures of the three angular (f1, f2, f3)
        * features for a given point based on its spatial neighborhood of 3D points with normals
        * \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
        * \param[in] normals the dataset containing the surface normals at each point in \a cloud
        * \param[in] indices the k-neighborhood point indices in the dataset
        * \param[in] nr_split the number of subdivisions for each angular feature interval
        * \param[out] pfh_histogram the resultant (combinatorial) PFH histogram representing the feature at the query point
        */
      void 
      computePointPFHSignature (const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals, 
                                const std::vector<int> &indices, int nr_split, Eigen::VectorXf &pfh_histogram);

    protected:
      /** \brief Estimate the Point Feature Histograms (PFH) descriptors at a set of points given by
        * <setInputCloud (), setIndices ()> using the surface in setSearchSurface () and the spatial locator in
        * setSearchMethod ()
        * \param[out] output the resultant point cloud model dataset that contains the PFH feature estimates
        */
      void 
      computeFeature (PointCloudOut &output);

      /** \brief The number of subdivisions for each angular feature interval. */
      int nr_subdiv_;

      /** \brief Placeholder for a point's PFH signature. */
      Eigen::VectorXf pfh_histogram_;

      /** \brief Placeholder for a PFH 4-tuple. */
      Eigen::Vector4f pfh_tuple_;

      /** \brief Placeholder for a histogram index. */
      int f_index_[3];

      /** \brief Float constant = 1.0 / (2.0 * M_PI) */
      float d_pi_; 
  };
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/features/impl/pfh.hpp>
#endif

#endif  //#ifndef PCL_PFH_H_

