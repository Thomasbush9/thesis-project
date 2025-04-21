from setuptools import setup, find_packages

setup(
    name='thesis_project',
    version='0.1',
    packages=find_packages(),  # This finds keyframe_selection, etc.
    py_modules=['utils'],      # Manually include top-level utils.py
)
