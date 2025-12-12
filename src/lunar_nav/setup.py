from setuptools import setup
from glob import glob
import os

package_name = 'lunar_nav'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'msg'), glob('msg/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Lunar crater navigation system',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crater_detector = lunar_nav.crater_detector:main',
            'particle_filter = lunar_nav.particle_filter:main',
            'teleop_keyboard = lunar_nav.teleop_keyboard:main',
            'navigator = lunar_nav.navigator:main',
        ],
    },
)
