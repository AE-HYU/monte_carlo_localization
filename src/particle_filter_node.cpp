#include "particle_filter_cpp/particle_filter.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<particle_filter_cpp::ParticleFilter>();
        
        RCLCPP_INFO(node->get_logger(), "Particle Filter Node started successfully");
        
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        std::cerr << "Exception in particle filter node: " << e.what() << std::endl;
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}