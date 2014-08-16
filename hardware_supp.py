__author__ = 'samyvilar'

import subprocess
import platform


def mac_avx_supported():
    return ' AVX1.0 ' in subprocess.check_output(('sysctl', '-a', 'machdep.cpu.features'))


def linux_avx_supported():
    return ' avx ' in subprocess.check_output(('cat', '/proc/cpuinfo'))


def win_avx_supported():
    raise NotImplementedError


default_impls = {'Darwin': mac_avx_supported, 'Linux': linux_avx_supported, 'Windows': win_avx_supported}


def avx_supported(impls=default_impls):
    return impls[platform.system()]()

# SSE is widely supported ...
supported_intrinsics = ('', 'sse') + ('avx',) * avx_supported()


