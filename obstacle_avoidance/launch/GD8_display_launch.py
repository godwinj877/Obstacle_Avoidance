# !usr/bin/env/python3

import os
import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros

def generate_launch_description():
    package_name = 'obstacle_avoidance'

    package_path = launch_ros.substitutions.FindPackageShare(package=package_name).find('obstacle_avoidance')
    default_model_path = os.path.join(package_path, 'urdf/urdf/Robot.xacro')
    default_rviz_config_path = os.path.join(package_path, 'rviz/GD8.rviz')

    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description' : Command(['xacro ', LaunchConfiguration('model')])}]
    )
    
    joint_state_publisher_node = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
    )
    
    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
    )

    spawn_entity = launch_ros.actions.Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'GD8', '-topic', 'robot_description'],
        output = 'screen'
    )

    robot_localization_node = launch_ros.actions.Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[os.path.join(package_path, 'config/ekf.yaml'), {'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    world_path = os.path.join(package_path, 'world/room1.world')

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(name='use_sim_time', default_value='True', 
                                             description='Flag to enable use_sim_time'),
        launch.actions.DeclareLaunchArgument(name='model', default_value=default_model_path,
                                             description='Absolute path to robot urdf file'),
        # launch.actions.DeclareLaunchArgument(name='rvizconfig', default_value=default_rviz_config_path,
        #                                      description='Absolute path to rviz config file'),
        launch.actions.ExecuteProcess(cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', world_path],
                                      output='screen'),
        joint_state_publisher_node,
        robot_state_publisher_node, 
        spawn_entity,
        # robot_localization_node,
        # rviz_node
    ])
