/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2015, Michael 'v4hn' Goerner
 *  Copyright (c) 2015-, Open Perception, Inc.
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
 */

#ifndef PCL_REGISTRATION_INCREMENTAL_ICP_H_
#define PCL_REGISTRATION_INCREMENTAL_ICP_H_

#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

namespace pcl {
  namespace registration {

    /** \brief Incremental @ref IterativeClosestPoint class
      *
      * This class provides a way to register a stream of clouds where each cloud will be aligned to the previous cloud.
      *
      * \code
      * IterativeClosestPoint<PointXYZ,PointXYZ>::Ptr icp (new IterativeClosestPoint<PointXYZ,PointXYZ>);
      * icp->setMaxCorrespondenceDistance (0.05);
      * icp->setMaximumIterations (50);
      *
      * IncrementalICP<PointXYZ> iicp;
      * iicp.setICP (icp);
      *
      * while (true){
      *   PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>);
      *   read_cloud (*cloud);
      *   iicp.registerCloud (cloud);
      *
      *   PointCloud<PointXYZ>::Ptr tmp (new PointCloud<PointCloudXY>);
      *   transformPointCloud (*cloud, *tmp, iicp.getAbsoluteTransform ());
      *   write_cloud (*tmp);
      * }
      * \endcode
      *
      * \author Michael 'v4hn' Goerner
      * \ingroup registration
      */
    template <typename PointT, typename Scalar = float>
    class IncrementalICP {
      public:
        typedef typename pcl::PointCloud<PointT>::Ptr PointCloudPtr;
        typedef typename pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;

        typedef typename pcl::Registration<PointT,PointT,Scalar>::Ptr RegistrationPtr;
        typedef typename pcl::Registration<PointT,PointT,Scalar>::Matrix4 Matrix4;

        IncrementalICP ();

        /** \brief Empty destructor */
        virtual ~IncrementalICP () {}

        /** \brief Register new point cloud incrementally
          * \note You have to set a valid registration object with @ref setICP before using this
          * \note The class doesn't copy cloud. If you afterwards change cloud, that will affect this class.
          * \param[in] cloud point cloud to register
          * \param[in] delta_estimate estimated transform between last registered cloud and this one
          * \return true if ICP converged
          */
        bool
        registerCloud (const PointCloudConstPtr& cloud, const Matrix4& delta_estimate = Matrix4::Identity ());

        /** \brief Get estimated transform between the last two registered clouds */
        inline Matrix4
        getDeltaTransform () const;

        /** \brief Get estimated overall transform */
        inline Matrix4
        getAbsoluteTransform () const;

        /** \brief Reset incremental ICP without resetting icp_ */
        inline void
        reset ();

        /** \brief Set registration instance used to align clouds */
        inline void
        setICP (RegistrationPtr);
      protected:

        /** \brief last registered point cloud */
        PointCloudConstPtr last_cloud_;

        /** \brief registration instance to align clouds */
        RegistrationPtr icp_;

        /** \brief estimated transforms */
        Matrix4 delta_transform_;
        Matrix4 abs_transform_;
    };

  }
}

#include <pcl/registration/impl/incremental_icp.hpp>

#endif /*PCL_REGISTRATION_INCREMENTAL_ICP_H_*/
