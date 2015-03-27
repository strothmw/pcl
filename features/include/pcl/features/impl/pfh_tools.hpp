

#ifndef _PCL_PFH_TOOL_IMPL_HPP_
#define _PCL_PFH_TOOL_IMPL_HPP_


#include <pcl/features/pfh.h>

//////////////////////////////////////////////////////////////////////////////////////////////

template <typename PointInT, typename PointNT>
bool pcl::PFHPairFeaturesManagedCache< PointInT, PointNT >::computePairFeatures
		    (
			  const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals, 
			  int p_idx, int q_idx, float &f1, float &f2, float &f3, float &f4
		    )
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

template <typename PointInT, typename PointNT>
bool pcl::PFHPairFeaturesManagedCache< PointInT, PointNT >::getPairFeatures
		    (
			  const pcl::PointCloud<PointInT> &cloud, const pcl::PointCloud<PointNT> &normals, 
			  int p_idx, int q_idx, Eigen::Vector4f &ar_pfh_tuple, 
			  bool ao_optimize_for_organized
		    )
{
  bool lo_return;
  
   static int lo_found_sum;
   static int lo_not_found_sum;
  
  if (use_cache_)
  {
    // In order to create the key, always use the bigger index as the first key pair member
      
    if ( ao_optimize_for_organized )
    {
      // always true, at least for organized cloud and OrganizedNeighbor search method
      // optimized out in release
      assert( q_idx < p_idx );
    }
    else
    {
      int lo_p1_swap = std::max( p_idx, q_idx ), lo_p2_swap = std::min( p_idx, q_idx );
      p_idx = lo_p1_swap;
      q_idx = lo_p2_swap;
    }
    
   ROS_ASSERT( q_idx < p_idx );
    
    IdFeatureMapType& lr_id_map = pv_feature_maps[ p_idx ];
    
    std::pair<IdFeatureMapType::iterator, bool> lo_insert_return;
    
    {
      // a "pseudo" feature map obj to test if it can be inserted to the map
      // if it can be inserted the value is not yet cached and must be
      // calculated
      std::pair<int, FeatureMapObj> lo_pair;
      lo_pair.first = q_idx;
      
      lo_insert_return = lr_id_map.insert( lo_pair );
    }
    
    if ( lo_insert_return.second )
    {
   lo_not_found_sum++;
      
      // inserting into feature map succeeded
      // the "pseudo" value was inserted
      //   i. e. it was not found in the cache
      // we have to calculate it and return the newly calculated value
      
      lo_return = computePairFeatures
			( 	
			    cloud, normals, p_idx, q_idx,
			    ar_pfh_tuple[0], ar_pfh_tuple[1], ar_pfh_tuple[2], ar_pfh_tuple[3] 
			);
			
      // and assign the newly calculated value to the returned iterator, thereby 
      // "really" filling the inserted field
			
      lo_insert_return.first->second.first = ar_pfh_tuple;
      lo_insert_return.first->second.second = lo_return;
    }
    else
    {
      lo_found_sum++;
      
      // inserting into feature map did not succeed
      // the "pseudo" value was not inserted
      //   i. e. a value for the key it was not found in the cache
      //   the value found the the cache is pointed by the returned iterator
      // we have to return the value pointed by the iterator
      
      ar_pfh_tuple = lo_insert_return.first->second.first ;
      lo_return = lo_insert_return.first->second.second ;
   	
       // Use a maximum cache so that we don't go overboard on RAM usage
//       key_list_.push (lo_key);
//       // Check to see if we need to remove an element due to exceeding max_size
//       if (key_list_.size () > max_cache_size_)
//       {
// 	// Remove the last element.
// 	feature_map_.erase (key_list_.front() );
// 	key_list_.pop ();
//       }
    }
  }
  else
  {
    lo_return = computePairFeatures
		      ( 	
			  cloud, normals, p_idx, q_idx,
			  ar_pfh_tuple[0], ar_pfh_tuple[1], ar_pfh_tuple[2], ar_pfh_tuple[3] 
		      );
  }
  
  ROS_INFO_STREAM_THROTTLE( 0.25, "Calls: " << lo_found_sum + lo_not_found_sum );
  ROS_INFO_STREAM_THROTTLE( 0.25, "Found ratio: " << static_cast<double>( lo_found_sum ) / (static_cast<double>( lo_found_sum ) + lo_not_found_sum ) );
  // ROS_INFO_STREAM_THROTTLE( 2.5, "Cache size: " << key_list_.size() );
  // ROS_INFO_STREAM_THROTTLE( 2.5, "Max cache size: " << max_cache_size_ );
   
  return lo_return;
}

//////////////////////////////////////////////////////////////////////////////////////////////


#endif //def _PCL_PFH_TOOL_IMPL_HPP_