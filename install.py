from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop
from distutils import log
import os
import subprocess
import sys

new_path = '/'.join(sys.executable.split('/')[:-1])
current_path = os.environ['PATH']
modified_env = {'PATH': f'{new_path}{os.pathsep}{current_path}'}


class AWEInstall(_install):
    '''This will download lexicons required for AWE to run, such as en_core_web_sm and en_core_web_lg.
    Note that this operation will take some (bandwidth-dependent) time to complete.
    '''
    def run(self):
        _install.run(self)
        subprocess.run(['python', 'awe_components/setup/data.py'], env=modified_env)
