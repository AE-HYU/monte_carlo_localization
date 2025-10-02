# Code Refactoring Plan

## Goal
Split particle_filter.cpp (1685 lines) into modular components for better maintainability.

## Current Status
✅ parameter_manager.hpp/cpp created (284 lines)
✅ map_manager.hpp/cpp created (197 lines)  
⏳ Need to create:
- initialization.hpp/cpp
- visualization.hpp/cpp  
- Update utils.cpp with coordinate transform functions
- Refactor particle_filter.cpp to use new modules
- Update CMakeLists.txt

## Module Breakdown

### parameter_manager (DONE)
- initParameters()
- dynamicParametersCallback()

### map_manager (DONE)  
- get_omap()
- try_load_map()
- precompute_sensor_model()

### initialization (TODO)
- initialize_global() 
- initialize_particles_pose()

### visualization (TODO)
- publish_tf()
- visualize()
- publish_particles()

### utils.cpp additions (TODO)
- apply_tf_offset()
- calculate_lidar_frame_motion()
- is_pose_valid()

### particle_filter.cpp (TODO - refactor)
Keep only:
- Constructor/Destructor
- MCL core (motion_model, sensor_model, resampling)
- Timer update
- Callbacks (lidar, odom, clicked_pose/point)

## Integration Steps
1. Create remaining module files
2. Make ParticleFilter members public/friend for module access
3. Update includes in particle_filter.hpp  
4. Replace function implementations with namespace calls
5. Update CMakeLists.txt to compile new sources
6. Build and test

## Expected Result
- parameter_manager.cpp: ~280 lines ✅
- map_manager.cpp: ~200 lines ✅
- initialization.cpp: ~150 lines
- visualization.cpp: ~180 lines
- utils.cpp: ~300 lines (expanded)
- particle_filter.cpp: ~500 lines (reduced from 1685)

Total: Better organized, easier to maintain!
