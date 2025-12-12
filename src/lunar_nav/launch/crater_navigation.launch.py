import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('lunar_nav')
    
    use_pf_arg = DeclareLaunchArgument(
        'use_pf', default_value='False',
        description='Use Particle Filter for Navigation Feedback'
    )
    world_file = os.path.join(pkg_share, 'worlds', 'lunar_crater.world')
    urdf_file = os.path.join(pkg_share, 'urdf', 'rover.urdf.xacro')
    
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={'world': world_file}.items()
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': ParameterValue(
            Command(['xacro ', urdf_file]), value_type=str)}]
    )
    
    # Spawn entity
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'lunar_rover', '-x', '-10.0', '-y', '-10.0', '-z', '1.0'],
        output='screen'
    )
    
    # Crater detector
    crater_detector = Node(
        package='lunar_nav',
        executable='crater_detector',
        name='crater_detector',
        parameters=[{
            'min_radius': 5,
            'max_radius': 100,
            'min_distance': 20,
            'canny_threshold': 30,
            'accumulator_threshold': 15
        }],
        output='screen'
    )
    
    # Particle filter
    particle_filter = Node(
        package='lunar_nav',
        executable='particle_filter',
        name='particle_filter',
        parameters=[{
            'n_particles': 100,
            'resample_threshold': 0.5,
            'map_size': 60.0,
            'alpha1': 0.05,
            'alpha2': 0.05,
            'alpha3': 0.05,
            'alpha4': 0.05,
            'sigma_r': 0.3,
            'sigma_d': 0.5,
            'initial_x': -10.0,
            'initial_y': -10.0
        }],
        output='screen'
    )
    
    navigator_node = Node(
        package='lunar_nav',
        executable='navigator',
        name='navigator',
        parameters=[{'use_pf': LaunchConfiguration('use_pf')}],
        output='screen'
    )
    
    # Static TF for map -> odom (Identity for now)
    map_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    return LaunchDescription([
        use_pf_arg,
        gazebo,
        robot_state_publisher,
        spawn_entity,
        crater_detector,
        particle_filter,
        navigator_node,
        map_tf
    ])
