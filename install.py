"""ComfyUI-Manager post-install hook for 3DA Scene.

- OpenCV only if no cv2 is importable (avoid duplicate opencv distributions).
- Python 3.13 only: bpy==5.2.3 wheel (GPL-3.0) with --no-deps, plus zstandard and cattrs.
  numpy/torch are never touched. bpy is used only in a separate worker process.
"""
import importlib.util
import subprocess
import sys


def pip(*args):
    return subprocess.call([sys.executable, '-m', 'pip', 'install', *args])


def main():
    if importlib.util.find_spec('cv2') is None:
        pip('opencv-python-headless')
    if sys.version_info[:2] == (3, 13) and importlib.util.find_spec('bpy') is None:
        if pip('--no-deps', 'bpy==5.2.3') == 0:
            pip('zstandard', 'cattrs')
        else:
            print('[3DA] bpy install failed; set blender_path in config/3da_server.yaml', file=sys.stderr)


if __name__ == '__main__':
    main()
