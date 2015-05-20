/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
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
 */

#ifndef PCL_FEATURES_PFH_TOOLS_H_
#define PCL_FEATURES_PFH_TOOLS_H_

#if defined __GNUC__
#  pragma GCC system_header 
#endif

#include <pcl/pcl_exports.h>
#include <Eigen/Core>

#include <boost/shared_ptr.hpp>
#include <boost/concept_check.hpp>

#include <map>
#include <vector>
#include <queue>
#include <unordered_map>

#include <pcl/point_cloud.h>

namespace pcl
{
  /** \brief Compute the 4-tuple representation containing the three angles and one distance between two points
    * represented by Cartesian coordinates and normals.
    * \note For explanations about the features, please see the literature mentioned above (the order of the
    * features might be different).
    * \param[in] p1 the first XYZ point
    * \param[in] n1 the first surface normal
    * \param[in] p2 the second XYZ point
    * \param[in] n2 the second surface normal
    * \param[out] f1 the first angular feature (angle between the projection of nq_idx and u)
    * \param[out] f2 the second angular feature (angle between nq_idx and v)
    * \param[out] f3 the third angular feature (angle between np_idx and |p_idx - q_idx|)
    * \param[out] f4 the distance feature (p_idx - q_idx)
    *
    * \note For efficiency reasons, we assume that the point data passed to the method is finite.
    * \ingroup features
    */
  PCL_EXPORTS bool 
  computePairFeatures (const Eigen::Vector4f &p1, const Eigen::Vector4f &n1, 
                       const Eigen::Vector4f &p2, const Eigen::Vector4f &n2, 
                       float &f1, float &f2, float &f3, float &f4);

  PCL_EXPORTS bool
  computeRGBPairFeatures (const Eigen::Vector4f &p1, const Eigen::Vector4f &n1, const Eigen::Vector4i &colors1,
                          const Eigen::Vector4f &p2, const Eigen::Vector4f &n2, const Eigen::Vector4i &colors2,
                          float &f1, float &f2, float &f3, float &f4, float &f5, float &f6, float &f7);

  
  template<typename PointInT, typename PointNT>
  class PFHPairFeaturesManagedCache
  {
  private:
      /** \brief Compute the 4-tuple representation containing the three angles and one distance between two points
        * represented by Cartesian coordinates and normals.
        * \note For explanations about the features, please see the literature mentioned above (the order of the
        * features might be different).
        * \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
        * \param[in] normals the dataset containing the surface normals (assuming normalized vectors) at each point in cloud
        * \param[in] p_idx the index of the first point (source)
        * \param[in] q_idx the index of the second point (target)
        * \param[out] f1 the first angular feature (angle between the projection of nq_idx and u)
        * \param[out] f2 the second angular feature (angle between nq_idx and v)
        * \param[out] f3 the third angular feature (angle between np_idx and |p_idx - q_idx|)
        * \param[out] f4 the distance feature (p_idx - q_idx)
        * \note For efficiency reasons, we assume that the point data passed to the method is finite.
        */
      bool 
      computePairFeatures(const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals, 
				    int p_idx, int q_idx, float &f1, float &f2, float &f3, float &f4);
      
  protected:
    
      /** \brief check if the feature for this pair is yet in cache, not call the compute function, if is return cached value
        * \note For explanations about the features, please see the literature mentioned above (the order of the
        * features might be different).
        * \param[in] cloud the dataset containing the XYZ Cartesian coordinates of the two points
        * \param[in] normals the dataset containing the surface normals (assuming normalized vectors) at each point in cloud
        * \param[in] p_idx the index of the first point (source)
        * \param[in] q_idx the index of the second point (target)
        * \param[out] f1 the first angular feature (angle between the projection of nq_idx and u)
        * \param[out] f2 the second angular feature (angle between nq_idx and v)
        * \param[out] f3 the third angular feature (angle between np_idx and |p_idx - q_idx|)
        * \param[out] f4 the distance feature (p_idx - q_idx)
        * \note For efficiency reasons, we assume that the point data passed to the method is finite.
        */
      bool 
      getPairFeatures (const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals, 
                           int p_idx, int q_idx, Eigen::Vector4f &ar_pfh_tuple, bool ao_optimize_for_organized = false );

      
      void initCache( size_t ao_length )
      {
	if ( use_cache_ )
	{
	  clearCache();
	  pv_feature_maps.resize( ao_length );
	}
      }
      
      void clearCache()
      {
	// Clear the feature map
	pv_feature_maps.clear ();
	std::queue<uint64_t> empty;
	std::swap (key_list_, empty);
      }
          
  public:

      PFHPairFeaturesManagedCache() :
        pv_feature_maps (),
        key_list_ (),
        // Default 1GB memory size. Need to set it to something more conservative.
        max_cache_size_ ((1ul*1024ul*1024ul*1024ul) / sizeof (std::pair<std::pair<int, int>, Eigen::Vector4f>)),
        use_cache_ (false)
      {
	
      }
	
      /** \brief Set the maximum internal cache size. Defaults to 2GB worth of entries.
        * \param[in] cache_size maximum cache size 
        */
      inline void
      setMaximumCacheSize (unsigned int cache_size)
      {
        max_cache_size_ = cache_size;
      }

      /** \brief Get the maximum internal cache size. */
      inline unsigned int 
      getMaximumCacheSize ()
      {
        return (max_cache_size_);
      }

      /** \brief Set whether to use an internal cache mechanism for removing redundant calculations or not. 
        *
        * \note Depending on how the point cloud is ordered and how the nearest
        * neighbors are estimated, using a cache could have a positive or a
        * negative influence. Please test with and without a cache on your
        * data, and choose whatever works best!
        *
        * See \ref setMaximumCacheSize for setting the maximum cache size
        *
        * \param[in] use_cache set to true to use the internal cache, false otherwise
        */
      inline void
      setUseInternalCache (bool use_cache)
      {
        use_cache_ = use_cache;
      }

      /** \brief Get whether the internal cache is used or not for computing the PFH features. */
      inline bool
      getUseInternalCache ()
      {
        return (use_cache_);
      }
      
  private:
      
      class FeatureMapObj
      {
      public:
	Eigen::Vector4f first;
	bool second;
	bool filled;
	
	FeatureMapObj() : filled( false ) {}
      };
    
      //typedef std::pair<Eigen::Vector4f, bool> FeatureMapObj;
      
      typedef std::map< int, FeatureMapObj > IdFeatureMapType;
      
      std::vector< IdFeatureMapType > pv_feature_maps;
    
      /** \brief Queue of pairs saved, used to constrain memory usage. */
      std::queue<uint64_t> key_list_;

      /** \brief Maximum size of internal cache memory. */
      unsigned int max_cache_size_;

      /** \brief Set to true to use the internal cache for removing redundant computations. */
      bool use_cache_;
  };
}

#include <pcl/features/impl/pfh_tools.hpp>

#endif  //#ifndef PCL_FEATURES_PFH_TOOLS_H_

