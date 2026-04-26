from setuptools import find_packages, setup

package_name = 'offboard_commander'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sunsakis',
    maintainer_email='sunsakis@pm.me',
    description='Anchor-frame visual positioning + flow-based hover test for PX4 SITL.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hover_test = offboard_commander.hover_test:main',
            'preflight_gate = offboard_commander.preflight_gate:main',
            'displacement_estimator = offboard_commander.displacement_estimator:main',
            'anchor_estimator = offboard_commander.anchor_estimator:main',
            'anchor_ev_shim = offboard_commander.anchor_ev_shim:main',
        ],
    },
)
