# OLD REFERENCE - Original particle_filter.cpp

This file contains the original monolithic implementation (1685 lines).
Kept for reference during refactoring.

**New modular structure:**
- `monte_carlo_localization.cpp` - Core MCL algorithm only
- `parameter_manager.cpp` - Parameter handling
- `map_manager.cpp` - Map loading and sensor model
- `initialization.cpp` - Particle initialization
- `visualization.cpp` - TF and RViz publishing
- `utils.cpp` - Utility functions

---

