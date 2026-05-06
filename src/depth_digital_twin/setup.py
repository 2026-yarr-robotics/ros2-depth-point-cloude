from glob import glob
from setuptools import find_packages, setup

package_name = 'depth_digital_twin'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='EunwooSong',
    maintainer_email='realityssu@gmail.com',
    description='RealSense + YOLO segmentation digital twin (point cloud + convex hull).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'capture_chessboard = depth_digital_twin.capture_chessboard:main',
            'calibrate = depth_digital_twin.calibration:main',
            'world_origin_node = depth_digital_twin.world_origin_node:main',
            'detection_node = depth_digital_twin.detection_node:main',
            'point_cloud_node = depth_digital_twin.point_cloud_node:main',
        ],
    },
)
