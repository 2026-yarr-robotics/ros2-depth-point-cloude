import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'recode_sequence'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.npy')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='EunwooSong',
    maintainer_email='song200348@gmail.com',
    description='Record/replay exo+hand RGB-D + Doosan M0609 trajectory.',
    license='BSD',
    entry_points={
        'console_scripts': [
            'camera_id_tool = recode_sequence.camera_id_tool:main',
            'recorder_node = recode_sequence.recorder_node:main',
            'sequence_player_node = recode_sequence.sequence_player_node:main',
            'playback_control = recode_sequence.playback_control:main',
            'handeye_tf_node = recode_sequence.handeye_tf_node:main',
        ],
    },
)
