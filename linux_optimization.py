import os
import logging
import subprocess
import platform

logger = logging.getLogger(__name__)

def set_cpu_governor(governor="performance"):
    """CPUガバナーを設定してパフォーマンスを最適化"""
    if platform.system() != "Linux":
        logger.info("Skipping CPU governor setting on non-Linux system")
        return False

    try:
        cpu_count = os.cpu_count() or 1
        success = True
        
        for i in range(cpu_count):
            governor_path = f"/sys/devices/system/cpu/cpufreq/policy{i}/scaling_governor"
            if os.path.exists(governor_path):
                cmd = f"echo {governor} | sudo tee {governor_path}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"Failed to set governor for CPU{i}: {result.stderr}")
                    success = False
        
        return success
    except Exception as e:
        logger.error(f"Error setting CPU governor: {e}")
        return False

def optimize_process_priority(nice_value=-10):
    """プロセス優先度を設定"""
    if platform.system() != "Linux":
        return False
    
    try:
        os.nice(nice_value)
        logger.info(f"Process priority set to {nice_value}")
        return True
    except Exception as e:
        logger.warning(f"Failed to set process priority: {e}")
        return False

def optimize_camera(device_path="/dev/video0", width=640, height=480, fps=30):
    """V4L2カメラパラメータを最適化"""
    if platform.system() != "Linux":
        return False
    
    try:
        # カメラフォーマット設定
        cmd1 = f"v4l2-ctl -d {device_path} --set-fmt-video=width={width},height={height},pixelformat=MJPG"
        # フレームレート設定
        cmd2 = f"v4l2-ctl -d {device_path} --set-parm={fps}"
        # バッファサイズ最小化（遅延削減）
        cmd3 = f"v4l2-ctl -d {device_path} --set-ctrl=buffersize=1"
        
        for cmd, desc in [(cmd1, "format"), (cmd2, "framerate"), (cmd3, "buffer")]:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Failed to set camera {desc}: {result.stderr}")
        
        return True
    except Exception as e:
        logger.error(f"Error optimizing camera: {e}")
        return False

def disable_power_saving():
    """省電力機能を無効化してレイテンシを改善"""
    if platform.system() != "Linux":
        return False
    
    try:
        # USBオートサスペンドを無効化
        cmd = "echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend"
        subprocess.run(cmd, shell=True, check=False)
        
        # Wi-Fi省電力モード無効化（存在する場合）
        wifi_cmd = "sudo iw dev wlan0 set power_save off"
        subprocess.run(wifi_cmd, shell=True, check=False)
        
        return True
    except Exception as e:
        logger.error(f"Error disabling power saving features: {e}")
        return False

def optimize_linux_for_camera():
    """カメラ動作用にLinuxシステムを最適化する関数"""
    logger.info("Optimizing Linux system for camera operations")
    
    if platform.system() != "Linux":
        logger.info("Not a Linux system, skipping optimizations")
        return
    
    # CPU最適化
    set_cpu_governor()
    
    # プロセス優先度設定
    optimize_process_priority()
    
    # カメラ最適化
    optimize_camera()
    
    # 省電力機能無効化
    disable_power_saving()
    
    logger.info("Linux optimization completed")