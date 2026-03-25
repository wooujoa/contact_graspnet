from setuptools import find_packages, setup

package_name = 'contact_grasp_pkg'

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
    maintainer='jwg',
    maintainer_email='wjddnrud4487@kw.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'grasp_inference_node = contact_grasp_pkg.grasp_inference_node:main',
            'grasp_filter_inference_node = contact_grasp_pkg.grasp_filter_inference_node:main',
            'grasp_filter_inference_Robot_node = contact_grasp_pkg.grasp_filter_inference_Robot_node:main',
        ],
    },
)
