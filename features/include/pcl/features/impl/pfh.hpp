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
 */

#ifndef PCL_FEATURES_IMPL_PFH_H_
#define PCL_FEATURES_IMPL_PFH_H_

#include <pcl/features/pfh.h>

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT, bool T_optimize_for_organized> bool
pcl::PFHEstimation<PointInT, PointNT, PointOutT, T_optimize_for_organized>::computePairFeatures (
      const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals,
      int p_idx, int q_idx, float &f1, float &f2, float &f3, float &f4)
{  
#if !( NDEBUG )
  assert( isFinite( cloud.points[p_idx] ) );
  assert( isFinite( cloud.points[q_idx] ) );
  assert( isFinite( normals.points[p_idx] ) );
  assert( isFinite( normals.points[q_idx] ) );
#endif // !( NDEBUG )
  
  bool lo_return = pcl::computePairFeatures
			    (
				cloud.points[p_idx].getVector4fMap (), normals.points[p_idx].getNormalVector4fMap (),
			        cloud.points[q_idx].getVector4fMap (), normals.points[q_idx].getNormalVector4fMap (),
				f1, f2, f3, f4
			    );
  
#if !( NDEBUG )
  assert( std::isfinite( f1 ) );
  assert( std::isfinite( f2 ) );
  assert( std::isfinite( f3 ) );
  assert( std::isfinite( f4 ) );
  
  float lo_f1, lo_f2, lo_f3, lo_f4;
      
  bool lo_return2 = pcl::computePairFeatures
			    (
			        cloud.points[q_idx].getVector4fMap (), normals.points[q_idx].getNormalVector4fMap (),
				cloud.points[p_idx].getVector4fMap (), normals.points[p_idx].getNormalVector4fMap (),
				lo_f1, lo_f2, lo_f3, lo_f4
			    );
    
  assert( std::isfinite( lo_f1 ) );
  assert( std::isfinite( lo_f2 ) );
  assert( std::isfinite( lo_f3 ) );
  assert( std::isfinite( lo_f4 ) );

  // these assertions, first of all on f3 can fail without the additions to pfh_tools of 26.03.2015
  assert( std::abs( lo_f1 - f1 ) < 0.000001 );
  assert( std::abs( lo_f2 - f2 ) < 0.000001 );
  assert( std::abs( lo_f3 - f3 ) < 0.000001 );
  assert( std::abs( lo_f4 - f4 ) < 0.000001 );

  return lo_return && lo_return2;
#else
  return lo_return;
#endif // !( NDEBUG )
  
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT, bool T_optimize_for_organized> void
pcl::PFHEstimation<PointInT, PointNT, PointOutT, T_optimize_for_organized>::computePointPFHSignature (
      const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals,
      const std::vector<int> &indices, int nr_split, Eigen::VectorXf &pfh_histogram)
{
  int h_index, h_p;

  // Clear the resultant point histogram
  pfh_histogram.setZero ();

  // Factorization constant
  float hist_incr = 100.0f / static_cast<float> (indices.size () * (indices.size () - 1) / 2);

  uint64_t key;
  bool key_found = false;
  
  // Iterate over all the points in the neighborhood
  for (size_t i_idx = 0; i_idx < indices.size (); ++i_idx)
  {
    for (size_t j_idx = 0; j_idx < i_idx; ++j_idx)
    {
      // If the 3D points are invalid, don't bother estimating, just continue
      // checking isFinite on normals rather than on points
      //	vaild points may have invalid normals (e. g. using IntegralImageNormalEstimation with BORDER_POLICY_IGNORE)
      //	but it's not possible to estimate normals for invalid points, so checking normals are vaild is safer!ws
      if (!isFinite (normals.points[indices[i_idx]]) || !isFinite (normals.points[indices[j_idx]]))
        continue;

      if (use_cache_)
      {
        // In order to create the key, always use the smaller index as the first key pair member
        uint32_t p1, p2;
        
	p1 = indices[i_idx];
        p2 = indices[j_idx];
	
	if ( T_optimize_for_organized )
	{
	  // always true, at least for organized cloud and OrganizedNeighbor search method
	  // optimized out in release
	  assert( p2 < p1 );
	}
	else
	{
	  uint32_t p1_swap = std::max( p1, p2 ), p2_swap = std::min( p1, p2 );
	  p1 = p1_swap;
	  p2 = p2_swap;
	}
	
	// workaround for fast pairing the values, just copy both to a bigger variable
	memcpy( &key, &p1, sizeof( uint32_t ) );
	memcpy( reinterpret_cast<char*>(&key) + sizeof( uint32_t ), &p2, sizeof( uint32_t ) );
	  
        // Check to see if we already estimated this pair in the global hashmap
	std::unordered_map< 
			  uint64_t, 
			  Eigen::Vector4f, 
			  std::hash< uint64_t >, 
			  std::equal_to< uint64_t >, 
			  Eigen::aligned_allocator<Eigen::Vector4f> 
			  >::iterator fm_it = feature_map_.find (key);
			  
        if (fm_it != feature_map_.end ())
	{
          pfh_tuple_ = fm_it->second;
	  key_found = true;
	}
        else
        {
          // Compute the pair NNi to NNj
          if (!computePairFeatures (cloud, normals, indices[i_idx], indices[j_idx],
                                    pfh_tuple_[0], pfh_tuple_[1], pfh_tuple_[2], pfh_tuple_[3]))
	  {
            continue;
	  }
	  
	  key_found = false;
        }
      }
      else
        if (!computePairFeatures (cloud, normals, indices[i_idx], indices[j_idx],
                                  pfh_tuple_[0], pfh_tuple_[1], pfh_tuple_[2], pfh_tuple_[3]))
          continue;

      // Normalize the f1, f2, f3 features and push them in the histogram
      f_index_[0] = static_cast<int> (floor (nr_split * ((pfh_tuple_[0] + M_PI) * d_pi_)));
      if (f_index_[0] < 0)         f_index_[0] = 0;
      if (f_index_[0] >= nr_split) f_index_[0] = nr_split - 1;

      f_index_[1] = static_cast<int> (floor (nr_split * ((pfh_tuple_[1] + 1.0) * 0.5)));
      if (f_index_[1] < 0)         f_index_[1] = 0;
      if (f_index_[1] >= nr_split) f_index_[1] = nr_split - 1;

      f_index_[2] = static_cast<int> (floor (nr_split * ((pfh_tuple_[2] + 1.0) * 0.5)));
      if (f_index_[2] < 0)         f_index_[2] = 0;
      if (f_index_[2] >= nr_split) f_index_[2] = nr_split - 1;

      // Copy into the histogram
      h_index = 0;
      h_p     = 1;
      for (int d = 0; d < 3; ++d)
      {
        h_index += h_p * f_index_[d];
        h_p     *= nr_split;
      }
      pfh_histogram[h_index] += hist_incr;

      if ( use_cache_ && !key_found )
      {	  
        // Save the value in the hashmap
        feature_map_[key] = pfh_tuple_;

        // Use a maximum cache so that we don't go overboard on RAM usage
        key_list_.push (key);
        // Check to see if we need to remove an element due to exceeding max_size
        if (key_list_.size () > max_cache_size_)
        {
          // Remove the last element.
          feature_map_.erase (key_list_.front() );
          key_list_.pop ();
        }
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointNT, typename PointOutT, bool T_optimize_for_organized> void
pcl::PFHEstimation<PointInT, PointNT, PointOutT, T_optimize_for_organized>::computeFeature (PointCloudOut &output)
{
  if ( T_optimize_for_organized && ( !input_->isOrganized() || !normals_->isOrganized() ) )
  {
       // invalid
    PCL_ERROR ("[pcl::PFHEstimation::computeFeature] Passed not organized cloud to estimator that" 
	       "is optimized for and only usable with organized clouds!\n") ;
    return;
  }
	
  // Clear the feature map
  feature_map_.clear ();
  std::queue<uint64_t> empty;
  std::swap (key_list_, empty);

  pfh_histogram_.setZero (nr_subdiv_ * nr_subdiv_ * nr_subdiv_);

  // Allocate enough space to hold the results
  // \note This resize is irrelevant for a radiusSearch ().
  std::vector<int> nn_indices (k_);
  std::vector<float> nn_dists (k_);

  output.is_dense = true;
  // Save a few cycles by not checking every point for NaN/Inf values if the cloud is set to dense
  if (input_->is_dense)
  {
    // Iterating over the entire index vector
    for (size_t idx = 0; idx < indices_->size (); ++idx)
    {
      if (this->searchForNeighbors ((*indices_)[idx], search_parameter_, nn_indices, nn_dists) == 0)
      {
        for (int d = 0; d < pfh_histogram_.size (); ++d)
          output.points[idx].histogram[d] = std::numeric_limits<float>::quiet_NaN ();

        output.is_dense = false;
        continue;
      }

      // Estimate the PFH signature at each patch
      computePointPFHSignature (*surface_, *normals_, nn_indices, nr_subdiv_, pfh_histogram_);

      // Copy into the resultant cloud
      for (int d = 0; d < pfh_histogram_.size (); ++d)
        output.points[idx].histogram[d] = pfh_histogram_[d];
    }
  }
  else
  {
    // Iterating over the entire index vector
    for (size_t idx = 0; idx < indices_->size (); ++idx)
    {
      // checking isFinite on normals rather than on points
      //	vaild points may have invalid normals (e. g. using IntegralImageNormalEstimation with BORDER_POLICY_IGNORE)
      //	but it's not possible to estimate normals for invalid points, so checking normals are vaild is safer!ws
      if (!isFinite ((*normals_)[(*indices_)[idx]]) ||
          this->searchForNeighbors ((*indices_)[idx], search_parameter_, nn_indices, nn_dists) == 0)
      {
        for (int d = 0; d < pfh_histogram_.size (); ++d)
          output.points[idx].histogram[d] = std::numeric_limits<float>::quiet_NaN ();

        output.is_dense = false;
        continue;
      }

      // Estimate the PFH signature at each patch
      computePointPFHSignature (*surface_, *normals_, nn_indices, nr_subdiv_, pfh_histogram_);

      // Copy into the resultant cloud
      for (int d = 0; d < pfh_histogram_.size (); ++d)
        output.points[idx].histogram[d] = pfh_histogram_[d];
    }
  }
}

#define PCL_INSTANTIATE_PFHEstimation(T,NT,OutT) template class PCL_EXPORTS pcl::PFHEstimation<T,NT,OutT>;

#endif    // PCL_FEATURES_IMPL_PFH_H_ 

