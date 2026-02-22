#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from ai_msgs.msg import PerceptionTargets
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Float32
from pyzbar.pyzbar import decode as pyzbar_decode
from pyzbar.pyzbar import ZBarSymbol
from sensor_msgs.msg import Range
import math
import os
import time
import re
import numpy as np
import serial
import subprocess
from collections import deque
import transforms3d.euler as tfe
import threading


LOBOT__FRAME_HEADER = 0x55
LOBOT_CMD_SERVO_MOVE = 3

class ImuAngleSubscriber(Node):
    def __init__(self):
        super().__init__('imu_angle_subscriber')
        # 串口互斥锁
        self.serial_lock = threading.Lock()
        
        # 串口初始化（带异常处理）
        try:
            self.serialHandle = serial.Serial("/dev/ttyUSB1", 9600, timeout=1)
            self.get_logger().info("串口连接成功：/dev/ttyUSB1")
        except serial.SerialException as e:
            self.get_logger().error(f"串口连接失败：{str(e)}，舵机控制功能将失效")
            self.serialHandle = None
        
        # 运动参数
        self.first_move = True
        self.base_linear_velocity = 4.2
        self.slow_linear_velocity = 4.2
        self.deceleration_threshold = 0.7
        # 居中对准参数（强化任务二居中）
        self.image_center_x = 320.0  # 图像中心X坐标
        self.align_threshold = 10.0  # 任务二居中误差容忍度（±10像素，更严格）
        self.align_speed = 0.04  # 微调速度降低，提高精度
        
        # 任务状态标记
        self.task_on1 = False
        self.task_on2 = True  # 任务二激活标记
        self.task2_qr_pose_adjusted = False  # 二维码姿态是否已调整
        self.task_on3 = False
        self.gochang1 = False
        self.gochang2_active = False
        self.left_raw = False
        self.star_move = True
        self.team_name_played = False
        self.last_qr_data = None
        # 二维码处理状态标记
        self.qr_processed = False
        self.qr_content_backup = ""  # 备份已识别的二维码内容
        self.qr_recognition_count = 0  # 连续识别计数，用于防抖
        self.qr_recognition_timeout = 15  # 二维码扫描超时时间（秒）
        self.qr_scan_start_time = None  # 二维码扫描开始时间

        # -------------------------- 任务2核心参数 --------------------------
        self.task2_step = 1  # 1:扫描二维码 2:处理风扇位 3:完成
        self.task2_fan_order = [1, 5, 2, 6, 3, 7, 4, 8]  # 自定义顺序
        self.task2_fan_index = 0  # 当前处理的顺序索引（从0开始）
        self.task2_view_pos = 0  # 当前识别位置（0:上方 1:下方）
        self.task2_qr_lines = []  # 存储8行二维码内容（苹果/梨子交替）
        self.task2_qr_scanned = False  # 二维码是否扫描完成
        self.task2_target_fruit = ""  # 当前风扇位目标果实（苹果/梨子）
        # 任务二距离参数（理论位置）
        self.task2_left_dist = [2.3 - i*0.5 for i in range(4)]  # 1-4号距离（理论值）
        self.task2_right_dist = [2 - i*0.5 for i in range(4)]  # 5-8号距离（理论值）
        self.task2_dist_tol = 0.05  # 距离误差容忍度
        self.task2_yolo_detected = False  # 是否检测到果树
        self.task2_fan_done = [False]*8  # 标记每个风扇位是否完成
        # 单树抓取次数控制
        self.task2_grab_count = [0]*8  # 索引0-7对应风扇位1-8
        self.task2_max_grabs = 2  # 每棵树最多抓2次
        # 位置回位参数（强制使用理论位置）
        self.task2_current_fan_pos = None  # 当前处理的风扇位
        self.task2_returning = False  # 是否处于回位状态
        self.task2_return_threshold = 0.05  # 回位距离误差容忍度（使用理论位置）
        # 目标识别优先级（只抓ap1/pe1）
        self.task2_priority_targets = {"ap1", "pe1"}  # 优先抓取的目标
        self.task2_upper_scan_complete = False  # 上层扫描是否完成
        # 新增：识别超时与重试参数
        self.task2_recognition_timeout = 3.0  # 单视角识别超时时间（秒）
        self.task2_max_retry = 1  # 单视角最大重试次数
        self.task2_current_retry = 0  # 当前重试次数
        # ----------------------------------------------------------------------

        # 其他参数保持不变...
        self.task2_fandnum = 0
        self.task2_star = False
        self.task2_go_qr = True
        self.task2_star_grap = False

        self.in3_left = {1,2,3,4}
        self.in3_mid = {5,6,7,8}
        self.in3_right = {9,10,11,12}
        self.on1 = {4,8,12}
        self.on2 = {3,7,11}
        self.on3 = {2,6,10}
        self.on4 = {1,5,9}      
        self.chinese_items = []
        self.numbers = []
        self.now_num = None
        self.now_name = None
        self.task3_step = 0
        self.name_right = False
        self.task3_go_qr = True
        self.task3_getqr = False
        self.task3_staryolov = False
        self.task3_place = 2
        self.task3_goline = False
        self.task3_goon = False
        self.arm_left = False
        self.arm_right = False
        self.l_rnum = 0
        self.r_lnum = 0
        self.task3_goline_num = 0

        self.task_all_done = False
        self.go_to_9_step = 0
        self.gochang2_step = 0
        self.gochang2_turn_finished = False
        self.reached_2_5m = False

        self.kp_angular = 0.75
        self.max_angular_velocity =0.6
        self.angle_threshold_stop = 0.5
        self.angle_threshold_start = 1.1
        self.goal_yaw = None
        self.linear_velocity = self.base_linear_velocity
        self.kd_angular = 0.3
        self.slowdown_threshold =5
        self.last_error = 0.0
        self.last_error_time = None
        self.is_goal_reached = True
        self.imu_received = False
        self.lidar_data = None
        self.star_jigang = True
        self.calibration_done = False
        self.imu_yaw_history = deque(maxlen=10)
        self.calibration_start_time = None
        self.CALIBRATION_DURATION = 4.0
        self.imu_zero_offset = 0.0
        self.current_imu_yaw = 0.0
        self.imu_available = False

        self.is_turning = False  # 是否正在执行转弯动作
        self.turn_target_reached = False  # 转弯目标是否达成
        
        self.enable_straight_calibration = False  # 强制关闭直线校准

        self.target = 0.75
        self.target_err = 0.1
        self.target_finsh = False
        self.gochang1_step = 1

        self.yolov_x = 0.0      
        self.yolov_y = 0.0
        self.yolov_name = ""
        self.belive = 0.0
        self.min_confidence = 0.65
        self.star_yolov = False
        self.vision_active = False

        self.fand_num = 0
        self.ripe_names = ["ch1", "on1", "pu1", "to1"]
        self.raw_names = ["ch2", "on2", "pu2", "to2"]
        self.bad_names = ["bad"]
        self.left_fans = {0, 2, 4, 6}
        self.right_fans = {1, 3, 5, 7}
        self.in_mid = False
        self.trage_fand = False
        self.ch1_num = 0
        self.to1_num = 0
        self.on1_num = 0
        self.pu1_num = 0
        self.bad_count = 0

        self.audio_config = {
            "audio_device": "plughw:1,0",
            "audio_paths": {
                "队名": "/root/audio_workspace/wav_files/name.wav",
                "苹果": "/root/audio_workspace/wav_files/apple.wav",
                "梨子": "/root/audio_workspace/wav_files/lizi.wav",
                "南瓜": "/root/audio_workspace/wav_files/nangua.wav",
                "西红柿": "/root/audio_workspace/wav_files/xihongshi.wav",
                "辣椒": "/root/audio_workspace/wav_files/lajiao.wav",
                "洋葱": "/root/audio_workspace/wav_files/yangcong.wav",
                "坏果": "/root/audio_workspace/wav_files/bad.wav",
                "生南瓜": "/root/audio_workspace/wav_files/shengnangua.wav",
                "生西红柿": "/root/audio_workspace/wav_files/shengxihongshi.wav",
                "生辣椒": "/root/audio_workspace/wav_files/shenglajiao.wav",
                "生洋葱": "/root/audio_workspace/wav_files/shengyangcong.wav",
                "1": "/root/audio_workspace/wav_files/weizhi1.wav",
                "2": "/root/audio_workspace/wav_files/weizhi2.wav",
                "3": "/root/audio_workspace/wav_files/weizhi3.wav",
                "4": "/root/audio_workspace/wav_files/weizhi4.wav",
                "5": "/root/audio_workspace/wav_files/weizhi5.wav",
                "6": "/root/audio_workspace/wav_files/weizhi6.wav",
                "7": "/root/audio_workspace/wav_files/weizhi7.wav",
                "8": "/root/audio_workspace/wav_files/weizhi8.wav",
                "9": "/root/audio_workspace/wav_files/weizhi9.wav",
                "10": "/root/audio_workspace/wav_files/weizhi10.wav",
                "11": "/root/audio_workspace/wav_files/weizhi11.wav",
                "12": "/root/audio_workspace/wav_files/weizhi12.wav",
                "0ge": "/root/audio_workspace/wav_files/0ge.wav",
                "1ge": "/root/audio_workspace/wav_files/1ge.wav",
                "2ge": "/root/audio_workspace/wav_files/2ge.wav",
                "3ge": "/root/audio_workspace/wav_files/3ge.wav",
                "4ge": "/root/audio_workspace/wav_files/4ge.wav",
                "5ge": "/root/audio_workspace/wav_files/5ge.wav",
                "完成": "/root/audio_workspace/wav_files/finish.wav",
            }
        }
        self.audio_process = None
        
        self.last_played_item = None
        self.played_item_cooldown = 3.0
        self.last_played_time = 0.0

        self.imu_subscription = self.create_subscription(Imu, 'imu/data_raw', self.imu_callback, 10)
        self.yolo_subscription = self.create_subscription(PerceptionTargets, '/hobot_dnn_detection', self.listener_callback, 2)
        self.image_subscription = self.create_subscription(CompressedImage, 'image', self.image_callback, 10)
        self.lidar_subscription = self.create_subscription(Range, '/laser', self.lidar_callback, 10)
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_goal)
        self.twist = Twist()

        self.get_logger().info("机器人节点初始化完成，等待IMU校准...")

    # -------------------------- 任务2核心方法 --------------------------
    def task2_process_qr(self, data):
        """处理任务2的8行二维码，提取1-8位的苹果/梨子目标"""
        lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
        
        if len(lines) != 8:
            self.get_logger().warn(f"任务2二维码格式错误！需8行，实际{len(lines)}行")
            time.sleep(1.5)
            return False
        
        valid_fruits = {"苹果", "梨子"}
        invalid_lines = [line for line in lines if line not in valid_fruits]
        if invalid_lines:
            self.get_logger().warn(f"二维码包含无效内容：{invalid_lines}，仅支持苹果/梨子")
            time.sleep(1.5)
            return False
        
        self.task2_qr_lines = lines
        self.task2_qr_scanned = True
        self.get_logger().info(f"任务2二维码扫描完成，目标序列：{self.task2_qr_lines}")
        
        for fruit in self.task2_qr_lines:
            self.play_local_wav(fruit)
            time.sleep(1.2)
        
        return True

    def task2_get_target_dist(self, fan_pos):
        """根据风扇位（1-8）获取理论目标距离"""
        if 1 <= fan_pos <= 4:  # 左侧风扇位（理论值）
            return self.task2_left_dist[fan_pos - 1]
        elif 5 <= fan_pos <= 8:  # 右侧风扇位（理论值）
            return self.task2_right_dist[fan_pos - 5]
        else:
            self.get_logger().error(f"无效风扇位：{fan_pos}，仅支持1-8")
            return None

    def task2_move_to_fan(self, fan_pos, is_return=False):
        """移动到指定风扇位的理论位置（不记忆实际位置）"""
        target_dist = self.task2_get_target_dist(fan_pos)
        if target_dist is None:
            return False

        if self.lidar_data is None:
            self.get_logger().warn("无激光雷达数据，无法移动到目标风扇位")
            return False
        
        # 回位和正常移动都使用理论位置和相同的误差容忍度
        dist_tol = self.task2_return_threshold if is_return else self.task2_dist_tol
        err = self.lidar_data - target_dist
        
        if abs(err) < dist_tol:
            self.stop_robot()
            self.get_logger().info(f"{'回位到' if is_return else '到达'}理论位置：{fan_pos}号树，目标{target_dist}m，实际{self.lidar_data:.2f}m")
            return True
        else:
            # 回位时速度降低，提高精度
            speed_factor = 0.5 if is_return else 0.6
            adjust_speed = self.base_linear_velocity * speed_factor
            if err > dist_tol:
                self.twist.linear.x = -adjust_speed  # 后退
            elif err < -dist_tol:
                self.twist.linear.x = adjust_speed   # 前进
            self.vel_publisher.publish(self.twist)
            return False

    def task2_return_to_fan(self):
        """返回当前风扇位的理论位置"""
        if self.task2_current_fan_pos is None:
            self.get_logger().warn("无当前风扇位信息，无法回位")
            return False
        
        self.get_logger().info(f"回位到{self.task2_current_fan_pos}号树理论位置")
        return self.task2_move_to_fan(self.task2_current_fan_pos, is_return=True)

    def task2_switch_view(self, view_pos):
        """切换识别位置（0：上方视角 1：下方视角），强制重置识别状态"""
        self.stop_robot()
        self.vision_active = True
        self.star_yolov = True
        self.task2_yolo_detected = False
        self.yolov_name = ""  # 清空上一视角的识别结果
        self.in_mid = False   # 重置居中状态
        self.task2_view_pos = view_pos  # 强制更新当前视角
        current_view = "上方" if view_pos == 0 else "下方"
        
        if view_pos == 0:
            # 上方视角
            if 1 <= self.task2_current_fan_pos <= 4:
                self.task2_left_prep_on()
            else:
                self.task2_right_prep_on()
        else:
            # 下方视角
            if 1 <= self.task2_current_fan_pos <= 4:
                self.task2_left_prep_under()
            else:
                self.task2_right_prep_under()
        
        self.get_logger().info(f"{self.task2_current_fan_pos}号树 - 切换到{current_view}视角（优先识别ap1/pe1）")
        time.sleep(1.5)  # 等待机械臂到位

    def task2_check_fruit(self):
        """只检测ap1/pe1，忽略其他目标"""
        if not self.yolov_name:
            return False
        
        # 只处理ap1/pe1
        if self.yolov_name not in self.task2_priority_targets:
            self.get_logger().info(f"忽略非目标{self.yolov_name}，只抓取ap1/pe1")
            return False
        
        # 当前风扇位索引（1-8 → 0-7）
        fan_idx = self.task2_current_fan_pos - 1
        # 超过最大抓取次数则直接返回
        if self.task2_grab_count[fan_idx] >= self.task2_max_grabs:
            self.get_logger().info(f"{self.task2_current_fan_pos}号树已抓取{self.task2_max_grabs}次，停止抓取")
            return False

        # 验证果实类型是否匹配
        target_fruit = self.task2_qr_lines[fan_idx]
        detected_type = "苹果" if self.yolov_name == "ap1" else "梨子" if self.yolov_name == "pe1" else "未知"
        if detected_type == target_fruit:
            self.task2_yolo_detected = True
            self.get_logger().info(f"检测到匹配目标{self.yolov_name}！目标{target_fruit}")
            self.play_local_wav(target_fruit)
            time.sleep(1)
            return True
        else:
            self.get_logger().info(f"类型不匹配！目标{target_fruit}，实际{detected_type}")
            return False

    def task2_main_logic(self):
        """任务二主逻辑：理论位置回位+精准居中+优先ap1/pe1+无识别时自动切换"""
        # 步骤1：扫描二维码
        if self.task2_step == 1:
            if not self.task2_qr_scanned:
                self.get_logger().info("任务2步骤1/3：扫描8行二维码中...")
                self.star_move = False
                if not self.task2_qr_pose_adjusted:
                    self.qr_poss()  # 调整摄像头姿态（仅执行一次）
                    self.task2_qr_pose_adjusted = True
                # 二维码扫描超时处理
                if self.qr_scan_start_time is None:
                    self.qr_scan_start_time = time.time()
                elif time.time() - self.qr_scan_start_time > self.qr_recognition_timeout:
                    self.get_logger().warn("二维码扫描超时，重新尝试...")
                    self.qr_scan_start_time = time.time()
            else:
                self.task2_qr_pose_adjusted = False
                self.task2_step = 2
                self.task2_fan_index = 0  # 重置顺序索引为0
                self.get_logger().info("任务2进入风扇位处理阶段，按顺序1→5→2→6→3→7→4→8处理")
                return

        # 步骤2：按自定义顺序遍历处理风扇位
        elif self.task2_step == 2:
            if not hasattr(self, 'task2_sub_step'):
                self.task2_sub_step = ""
            
            # 所有风扇位处理完成，进入步骤3
            if self.task2_fan_index >= len(self.task2_fan_order):
                self.task2_step = 3
                return
            
            # 当前处理的风扇位
            self.task2_current_fan_pos = self.task2_fan_order[self.task2_fan_index]
            fan_idx = self.task2_current_fan_pos - 1  # 转换为0-7索引

            # 检查是否已达最大抓取次数
            if self.task2_grab_count[fan_idx] >= self.task2_max_grabs:
                self.get_logger().info(f"{self.task2_current_fan_pos}号树已达最大抓取次数，跳过")
                self.task2_fan_index += 1
                self.task2_sub_step = ""
                self.task2_current_fan_pos = None
                self.task2_upper_scan_complete = False  # 重置上层扫描标记
                return

            # 子步骤A：移动到当前风扇位的理论位置
            if self.task2_sub_step == "":
                self.get_logger().info(f"任务2处理{self.task2_current_fan_pos}号树（步骤A：移动到理论位置）")
                if not self.task2_move_to_fan(self.task2_current_fan_pos):
                    return
                # 先处理上层视角
                self.task2_view_pos = 0
                self.task2_upper_scan_complete = False  # 重置上层扫描标记
                self.task2_current_retry = 0  # 重置重试次数
                self.task2_switch_view(self.task2_view_pos)
                self.task2_view_timer = time.time()
                self.task2_sub_step = "view_check"
                return

            # 子步骤B：视角识别（优先上层，失败则切换下层）
            if self.task2_sub_step == "view_check":
                current_view = "上方" if self.task2_view_pos == 0 else "下方"
                
                # 1. 先执行精准居中（任务二专用）
                if not self.in_mid:
                    self.grad_adiust()
                    return

                # 2. 居中完成后检查果实（只认ap1/pe1）
                if self.task2_check_fruit():
                    # 执行抓取动作
                    self.get_logger().info(f"执行{current_view}视角抓取{self.yolov_name}")
                    if self.task2_view_pos == 0:
                        self.task2_on_grad()
                    else:
                        self.task2_under_grad()
                    
                    # 抓取次数+1
                    self.task2_grab_count[fan_idx] += 1
                    self.get_logger().info(f"{self.task2_current_fan_pos}号树抓取次数：{self.task2_grab_count[fan_idx]}/{self.task2_max_grabs}")
                    
                    # 进入回位子步骤（回到理论位置）
                    self.task2_sub_step = "return_to_position"
                    self.task2_returning = True
                    self.task2_current_retry = 0  # 重置重试次数
                    return

                # 3. 无识别结果处理（核心优化部分）
                elapsed = time.time() - self.task2_view_timer
                if elapsed > self.task2_recognition_timeout:
                    self.get_logger().info(
                        f"{self.task2_current_fan_pos}号树{current_view}视角超时未识别（{elapsed:.1f}s），"
                        f"重试次数：{self.task2_current_retry}/{self.task2_max_retry}"
                    )
                    self.vision_active = False
                    self.star_yolov = False

                    # 3.1 重试机制：未达最大重试次数则重新激活识别
                    if self.task2_current_retry < self.task2_max_retry:
                        self.task2_current_retry += 1
                        self.task2_switch_view(self.task2_view_pos)  # 重新激活当前视角
                        self.task2_view_timer = time.time()  # 重置计时器
                        return

                    # 3.2 重试失败：切换视角或推进到下一步
                    self.task2_current_retry = 0  # 重置重试次数
                    if self.task2_view_pos == 0:
                        # 上层识别失败→切换到下层
                        self.get_logger().info(f"上层识别失败，切换到下层视角")
                        self.task2_view_pos = 1
                        self.task2_switch_view(self.task2_view_pos)
                        self.task2_view_timer = time.time()
                    else:
                        # 下层识别失败→结束当前树处理
                        self.get_logger().info(f"上下层均未识别到目标，结束{self.task2_current_fan_pos}号树处理")
                        self.task2_fan_index += 1
                        self.task2_sub_step = ""
                        self.task2_current_fan_pos = None
                        self.task2_upper_scan_complete = False
                return

            # 子步骤C：抓取后回位到理论位置
            if self.task2_sub_step == "return_to_position":
                # 定义current_view变量（修复变量未定义错误）
                current_view = "上方" if self.task2_view_pos == 0 else "下方"
                
                if self.task2_return_to_fan():
                    self.task2_returning = False
                    self.get_logger().info(f"{self.task2_current_fan_pos}号树回位到理论位置完成")
                    # 重新切换到当前视角，准备再次识别（未达最大次数）
                    self.task2_switch_view(self.task2_view_pos)
                    self.task2_view_timer = time.time()
                    self.in_mid = False  # 重置居中状态
                    
                    # 检查是否已达最大抓取次数
                    if self.task2_grab_count[fan_idx] >= self.task2_max_grabs:
                        self.get_logger().info(f"{self.task2_current_fan_pos}号树已达最大抓取次数")
                        # 切换到下一个视角
                        if self.task2_view_pos == 0:
                            self.task2_view_pos = 1
                            self.task2_switch_view(self.task2_view_pos)
                            self.task2_view_timer = time.time()
                            self.task2_sub_step = "view_check"
                        else:
                            self.task2_fan_index += 1
                            self.task2_sub_step = ""
                            self.task2_current_fan_pos = None
                            self.task2_upper_scan_complete = False
                    else:
                        self.get_logger().info(f"准备再次识别{self.task2_current_fan_pos}号树的{current_view}视角")
                        self.task2_sub_step = "view_check"
                return

        # 步骤3：任务2完成
        if self.task2_step == 3:
            self.stop_robot()
            self.last_servos()  # 舵机复位
            self.get_logger().info("任务2所有风扇位处理完成！")
            self.play_local_wav("完成")
            time.sleep(2)
            self.task_on2 = False

    # -------------------------- 目标居中逻辑（强化任务二） --------------------------
    def grad_adiust(self):
        """任务二专用精准居中逻辑，其他任务保持原有范围"""
        if self.trage_fand:
            # 任务二居中范围：X=310-330（±10像素，更严格）
            if self.task_on2 and self.task2_step == 2:
                if 310.0 <= self.yolov_x <= 330.0:
                    self.in_mid = True
                    self.stop_robot()
                    self.get_logger().info(f"任务二目标居中完成（X={self.yolov_x:.1f}，误差±10像素）")
                else:
                    self.in_mid = False
                    # 微调速度降低，避免超调
                    if self.yolov_x < 310.0:
                        self.twist.linear.x = -self.align_speed  # 左移
                    elif self.yolov_x > 330.0:
                        self.twist.linear.x = self.align_speed   # 右移
                    self.vel_publisher.publish(self.twist)
            else:
                # 其他任务保持原有范围
                if 230.0 <= self.yolov_x <= 410.0:
                    self.in_mid = True
                    self.stop_robot()
                else:
                    self.in_mid = False
                    if self.yolov_x < 230.0:
                        self.twist.linear.x = -self.align_speed
                    elif self.yolov_x > 410.0:
                        self.twist.linear.x = self.align_speed
                    self.vel_publisher.publish(self.twist)

    # -------------------------- 其他方法保持不变 --------------------------
    def stop_audio(self):
        if self.audio_process and self.audio_process.poll() is None:
            self.audio_process.terminate()
            try:
                self.audio_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.audio_process.kill()
            self.get_logger().info("已停止当前音频播放")
        self.audio_process = None

    def play_local_wav(self, audio_key):
        self.stop_audio()
        self.stop_robot()

        if audio_key not in self.audio_config["audio_paths"]:
            self.get_logger().error(f"音频键不存在：{audio_key}")
            return
        
        audio_path = self.audio_config["audio_paths"][audio_key]
        if not os.path.exists(audio_path):
            self.get_logger().error(f"音频文件缺失：{audio_path}")
            return
        if not audio_path.lower().endswith(".wav"):
            self.get_logger().error(f"非WAV格式文件：{audio_path}")
            return

        try:
            self.audio_process = subprocess.Popen(
                ["aplay", "-q", "-D", self.audio_config["audio_device"], audio_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.get_logger().info(f"播放音频：{audio_path}")
        except Exception as e:
            self.get_logger().error(f"播放异常：{str(e)}")

    def stop_robot(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.vel_publisher.publish(self.twist)
        time.sleep(0.1)

    def move_robot1(self):
        self.twist.linear.x = self.linear_velocity
        self.twist.angular.z = 0.0
        self.vel_publisher.publish(self.twist)

    def back_robot1(self):
        self.twist.linear.x = -self.linear_velocity
        self.twist.angular.z = 0.0
        self.vel_publisher.publish(self.twist)

    def move_robot2(self):
        self.twist.linear.x = 0.05
        self.twist.angular.z = 0.0
        self.vel_publisher.publish(self.twist)

    def back_robot2(self):
        self.twist.linear.x = -0.05
        self.twist.angular.z = 0.0
        self.vel_publisher.publish(self.twist)

    def setPWMServoMoveByArray(self, servos, servos_count, time):
        if not self.serialHandle or not self.serialHandle.isOpen():
            self.get_logger().error("串口未连接，无法控制舵机")
            return
        
        expected_length = servos_count * 2
        if len(servos) != expected_length:
            self.get_logger().error(f"舵机参数错误：预期{expected_length}个元素，实际{len(servos)}个")
            return
        
        buf = bytearray(b'\x55\x55')
        buf.append(servos_count * 3 + 5)
        buf.append(LOBOT_CMD_SERVO_MOVE)
        
        servos_count = max(1, min(servos_count, 254))
        buf.append(servos_count)
        
        time = max(100, min(time, 30000))
        buf.extend(time.to_bytes(2, 'little'))
        
        for i in range(servos_count):
            buf.append(servos[i * 2])
            pos = max(500, min(servos[i * 2 + 1], 2500))
            buf.extend(pos.to_bytes(2, 'little'))

        try:
            with self.serial_lock:
                self.serialHandle.flushInput()
                self.serialHandle.write(buf)
                self.serialHandle.flush()
            self.get_logger().debug(f"舵机指令发送成功（长度：{len(buf)}字节）")
        except Exception as e:
            self.get_logger().error(f"舵机指令发送失败：{str(e)}")

    def qr_poss(self):
        servos1 = [1, 2100, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos1, 4, 1100)
        time.sleep(1.5)
        
        servos2 = [3, 2100, 4, 1200, 9, 2150]
        self.setPWMServoMoveByArray(servos2, 3, 1100)
        time.sleep(1.5)

        self.vision_active = True

    def left_prep(self):
        servos1 = [1, 1075, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos1, 4, 1100)
        time.sleep(1.5)
        
        servos2 = [3, 1440, 4, 1930, 9, 1220]
        self.setPWMServoMoveByArray(servos2, 3, 1100)
        time.sleep(1.5)
        self.arm_left = True
        self.arm_right = False
        self.vision_active = True
        self.star_yolov = True

    def right_prep(self):
        servos1 = [1, 2100, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos1, 4, 1100)
        time.sleep(1.5)
        
        servos2 = [3, 1440, 4, 1930, 9, 1220]
        self.setPWMServoMoveByArray(servos2, 3, 1100)
        time.sleep(1.5)
        self.arm_left = False
        self.arm_right = True

        self.vision_active = True
        self.star_yolov = True

    def left_grad(self):
        servos = [3, 1080, 4, 1550, 9, 930]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.5)
        
        servos = [3, 1030, 4, 1235, 9, 1080]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.5)
        
        servos = [8, 2300]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(1)

        servos = [3,1600 , 4 , 600 ,9 ,2100 ,8 ,2100]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        servos = [1, 1600, 3,1600 , 4 , 600 ,9 ,2100]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        servos = [8 ,1500]
        self.setPWMServoMoveByArray(servos, 1, 800)  
        time.sleep(1)
        
        if self.now_name in self.audio_config["audio_paths"]:
            self.play_local_wav(self.now_name)
        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False

    def right_grad(self):
        servos = [3, 1080, 4, 1550, 9, 930]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.5)
        
        servos = [3, 1030, 4, 1235, 9, 1080]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.5)
        
        servos = [8, 2300]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(1)

        servos = [3,1600 , 4 , 600 ,9 ,2100 ,8 ,2100]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        servos = [1, 1600, 3,1600 , 4 , 600 ,9 ,2100]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        servos = [8 ,1500]
        self.setPWMServoMoveByArray(servos, 1, 800)  
        time.sleep(1)
        
        if self.now_name in self.audio_config["audio_paths"]:
            self.play_local_wav(self.now_name)
        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False
    
    def bad_grad(self):
        servos = [3, 1080, 4, 1550, 9, 930]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.5)
        
        servos = [3, 1030, 4, 1235, 9, 1045]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.5)
        
        servos = [8, 2200]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(1)

        servos2 = [3, 1500, 4, 1725, 9, 1805]
        self.setPWMServoMoveByArray(servos2, 3, 1100)
        time.sleep(1.5)

        servos = [8 ,1500]
        self.setPWMServoMoveByArray(servos, 1, 800)  
        time.sleep(1)

        servos = [3,1700 , 4 , 600 ,9 ,2100 ,8 ,1800]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        servos = [1, 1600, 3,1700 , 4 , 600 ,9 ,2100]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        if self.now_name in self.audio_config["audio_paths"]:
            self.play_local_wav(self.now_name)
        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False

    def left_no(self):
        servos = [1, 1110, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos, 4, 1000)
        time.sleep(1.5)
        
        servos = [1, 1600]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(1)
        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False

    def right_no(self):
        servos = [1, 2100, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos, 4, 1000)
        time.sleep(1.5)
        
        servos = [1, 1600]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(1)
        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False

    def task2_left_prep_on(self):
        servos1 = [1, 1100, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos1, 4, 1100)
        time.sleep(1.5)
        
        servos2 = [3, 2040, 4, 1575, 9, 1755]
        self.setPWMServoMoveByArray(servos2, 3, 1100)
        time.sleep(1.5)

        self.vision_active = True

    def task2_right_prep_on(self):
        servos1 = [1, 2100, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos1, 4, 1100)
        time.sleep(1.5)
        
        servos2 = [3, 2040, 4, 1575, 9, 1755]
        self.setPWMServoMoveByArray(servos2, 3, 1100)
        time.sleep(1.5)

        self.vision_active = True

    def task2_left_prep_under(self):
        servos1 = [1, 1100, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos1, 4, 1100)
        time.sleep(1.5)
        
        servos2 = [3, 2165, 4, 1885, 9, 1840]
        self.setPWMServoMoveByArray(servos2, 3, 1100)
        time.sleep(1.5)

        self.vision_active = True

    def task2_right_prep_under(self):
        servos1 = [1, 2100, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos1, 4, 1100)
        time.sleep(1.5)
        
        servos2 = [3, 2165, 4, 1885, 9, 1840]
        self.setPWMServoMoveByArray(servos2, 3, 1100)
        time.sleep(1.5)

        self.vision_active = True

    def task2_on_grad(self):
        servos = [3, 1080, 4, 1550, 9, 930]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.5)
        
        servos = [3, 1030, 4, 1235, 9, 1080]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.5)
        
        servos = [8, 2300]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(1)

        servos = [3,1600 , 4 , 600 ,9 ,2100 ,8 ,2100]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        servos = [1, 1600, 3,1600 , 4 , 600 ,9 ,2100]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        servos = [8 ,1500]
        self.setPWMServoMoveByArray(servos, 1, 800)  
        time.sleep(1)

        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False

    def task2_under_grad(self):
        servos = [3, 1080, 4, 1550, 9, 930]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.5)
        
        servos = [3, 1030, 4, 1235, 9, 1080]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.5)
        
        servos = [8, 2300]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(1)

        servos = [3,1600 , 4 , 600 ,9 ,2100 ,8 ,2100]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        servos = [1, 1600, 3,1600 , 4 , 600 ,9 ,2100]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        servos = [8 ,1500]
        self.setPWMServoMoveByArray(servos, 1, 800)  
        time.sleep(1)

        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False

    def last_servos(self):
        servos = [1, 1600, 3, 1700, 4, 600, 8, 1500, 9, 2100]
        self.setPWMServoMoveByArray(servos, 5, 1500)
        time.sleep(2)

    def lidar_callback(self, msg):
        if self.star_jigang:
            self.lidar_data = msg.range
        else:
            self.lidar_data = None

    def process_qr_data(self, data):
        if not data.strip():
            return [], []
            
        lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
        
        number_line_index = -1
        for i, line in enumerate(lines):
            if re.match(r'^[\d,]+$', line):
                number_line_index = i
                break
                
        if number_line_index != -1:
            chinese_part = lines[:number_line_index]
            numbers_part = lines[number_line_index]
        else:
            if len(lines) >= 1:
                chinese_part = lines[:-1]
                numbers_part = lines[-1]
            else:
                chinese_part = lines
                numbers_part = ""
                
        chinese_items = [item for item in chinese_part if item in ["洋葱", "南瓜", "西红柿", "辣椒"]]
        
        numbers = []
        if numbers_part:
            number_strings = re.split(r'[, ]+', numbers_part)
            for num_str in number_strings:
                if num_str.isdigit():
                    numbers.append(int(num_str))
        
        return chinese_items, numbers

    def normalize_angle(self, angle):
        return (angle + 180) % 360 - 180

    def set_new_goal(self, delta_deg):
        self.goal_yaw = self.normalize_angle(self.goal_yaw + delta_deg)
        self.is_goal_reached = False

    def go_target(self):
        if self.star_move:
            if not self.lidar_data:
                self.get_logger().warn("无激光雷达数据")
                return
            
            err = self.lidar_data - self.target
            
            if self.first_move:
                self.linear_velocity = self.slow_linear_velocity
            else:
                self.linear_velocity = self.base_linear_velocity

            if abs(err) < self.target_err:
                self.target_finsh = True
                self.stop_robot()
                if self.first_move:
                    self.first_move = False
                    self.get_logger().info("首次运动完成，切换到基础速度")
            else:
                self.target_finsh = False
                if err > self.target_err:
                    self.back_robot1()
                elif err < -self.target_err:
                    self.move_robot1()

    def gochang2(self):
        if self.gochang2_active:
            if self.gochang2_step == 0:
                self.get_logger().info("启动gochang2任务，初始化状态")
                self.reached_2_5m = False
                self.reached_2m_after_turn = False
                self.turn_step_done = False
                self.vehicle_stopped = False
                self.report_started = False
                self.report_complete = False
                self.target = 2.5
                self.initial_yaw_for_turn = None
                self.star_move = True
                self.get_logger().info("初始化完成，准备先前进到2.5米")
                self.gochang2_step = 1

            elif self.gochang2_step == 1:
                self.get_logger().info(f"gochang2步骤1/5：前往2.5米（当前距离：{self.lidar_data:.2f}米）")
                self.go_target()
                if self.target_finsh:
                    self.stop_robot()
                    time.sleep(1.5)
                    self.reached_2_5m = True
                    self.initial_yaw_for_turn = self.yaw_deg
                    self.get_logger().info(f"已到达2.5米，转弯基准角度：{self.initial_yaw_for_turn:.2f}°")
                    self.gochang2_step = 2

            elif self.gochang2_step == 2:
                self.get_logger().info("gochang2步骤2/5：执行右转90度")
                if not self.turn_step_done and self.initial_yaw_for_turn is not None:
                    target_absolute_yaw = self.normalize_angle(self.initial_yaw_for_turn + 90.0)
                    if not self.is_turning:
                        self.goal_yaw = target_absolute_yaw
                        self.is_turning = True
                        self.turn_target_reached = False
                        self.get_logger().info(f"开始右转：基准角度={self.initial_yaw_for_turn:.2f}°，目标={target_absolute_yaw:.2f}°")
                    
                    current_error = abs(self.normalize_angle(target_absolute_yaw - self.yaw_deg))
                    if current_error < self.angle_threshold_stop:
                        self.is_turning = False
                        self.turn_target_reached = True
                        self.turn_step_done = True
                        self.stop_robot()
                        time.sleep(1.5)
                        self.target = 2.0
                        self.target_finsh = False
                        self.get_logger().info(f"转弯完成（误差={current_error:.2f}°），准备前进2米")
                        self.gochang2_step = 3

            elif self.gochang2_step == 3:
                self.get_logger().info(f"gochang2步骤3/5：前往2米（当前距离：{self.lidar_data:.2f}米）")
                self.go_target()
                if self.target_finsh:
                    self.stop_robot()
                    time.sleep(2.0)
                    self.reached_2m_after_turn = True
                    self.vehicle_stopped = True
                    self.get_logger().info("已到达2米目标点，小车已强制停稳")
                    self.gochang2_step = 4

            elif self.gochang2_step == 4:
                if not self.vehicle_stopped:
                    self.get_logger().info("二次确认停车")
                    self.stop_robot()
                    time.sleep(1)
                    self.vehicle_stopped = True
                
                if not self.report_started:
                    self.get_logger().info("gochang2步骤4/5：开始播报采收情况")
                    self.report_started = True
                    fruit_count_map = [
                        ("南瓜", self.pu1_num),
                        ("辣椒", self.ch1_num),
                        ("西红柿", self.to1_num),
                        ("洋葱", self.on1_num)
                    ]
                    for fruit_name, count in fruit_count_map:
                        if fruit_name in self.audio_config["audio_paths"]:
                            self.play_local_wav(fruit_name)
                            time.sleep(1.2)
                        num_audio_key = f"{count}ge"
                        if num_audio_key in self.audio_config["audio_paths"]:
                            self.play_local_wav(num_audio_key)
                            time.sleep(1)
                    self.report_complete = True
                    self.get_logger().info("播报完成")
                    self.gochang2_step = 5

            elif self.gochang2_step == 5:
                self.get_logger().info("gochang2步骤5/5：彻底终止任务")
                self.stop_robot()
                self.stop_audio()
                self.star_move = False
                self.is_turning = False
                self.target_finsh = True
                self.gochang2_active = False
                self.gochang2_step = 0
                self.get_logger().info("全流程完成，小车保持静止")

    def gage_name(self):
        if self.task_on3:
            if self.now_name == "辣椒":
                self.name_right = (self.yolov_name == "ch1")
            elif self.now_name == "西红柿":
                self.name_right = (self.yolov_name == "to1")
            elif self.now_name == "洋葱":
                self.name_right = (self.yolov_name == "on1")
            elif self.now_name == "南瓜":
                self.name_right = (self.yolov_name == "pu1")
            elif self.now_name == "坏果":
                self.name_right = (self.yolov_name == "bad")
            
            if self.name_right and self.yolov_name:
                if self.arm_left:
                    self.left_grad()
                elif self.arm_right:
                    self.right_grad()
                if self.yolov_name == "ch1":
                    self.ch1_num += 1
                elif self.yolov_name == "to1":
                    self.to1_num += 1
                elif self.yolov_name == "on1":
                    self.on1_num += 1
                elif self.yolov_name == "pu1":
                    self.pu1_num += 1
            elif self.yolov_name in self.bad_names:
                self.bad_count += 1
                self.get_logger().info(f"任务3抓取坏果：{self.yolov_name}")
                if self.arm_left or self.arm_right:
                    self.bad_grad()
            else:
                self.get_logger().info(f"不匹配目标：{self.yolov_name}，不抓取")
                if self.arm_left:
                    self.left_no()
                elif self.arm_right:
                    self.right_no()
            
            self.task3_staryolov = False
            self.in_mid = False
            self.task3_goon = False
            self.task3_goline = False
            self.yolov_name = ""
            self.yolov_x = 0.0
            self.yolov_y = 0.0
            self.task3_step += 1
            self.star_move = True
            self.trage_fand = False

    def car_adjust(self):
        if not self.calibration_done:
            self.stop_robot()
            return

        if not self.imu_received:
            self.get_logger().warn("无IMU数据，无法进行角度调整")
            self.stop_robot()
            return

        if not self.is_turning:
            return

        angle_error = self.normalize_angle(self.goal_yaw - self.yaw_deg)
        
        if abs(angle_error) < self.angle_threshold_stop:
            if not self.is_goal_reached:
                self.stop_robot()
                self.is_goal_reached = True
                self.star_jigang = True  
                self.turn_target_reached = True
                self.get_logger().info(f"转弯完成，当前角度: {self.yaw_deg:.2f}°，目标角度: {self.goal_yaw:.2f}°")
            return
        elif abs(angle_error) > self.angle_threshold_start:
            self.is_goal_reached = False

        if not self.is_goal_reached:
            if self.task_on1:
                self.get_logger().info(f"任务1转弯调整 - 误差: {angle_error:.2f}°")
            elif self.task_on3:
                self.get_logger().info(f"任务3转弯调整 - 误差: {angle_error:.2f}°")
            elif self.gochang2_active:
                self.get_logger().info(f"最终阶段转弯调整 - 误差: {angle_error:.2f}°")

            self.star_jigang = False
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            if self.last_error_time is not None:
                dt = current_time - self.last_error_time
                error_derivative = (angle_error - self.last_error) / dt if dt > 0 else 0
            else:
                error_derivative = 0
            
            angular_velocity = (
                self.kp_angular * angle_error + 
                self.kd_angular * error_derivative
            )
            
            angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)
            
            if abs(angle_error) < self.slowdown_threshold:
                scale_factor = 0.05
                angular_velocity *= (abs(angle_error) / self.slowdown_threshold) * scale_factor
            
            min_angular_velocity = 0.1
            if abs(angular_velocity) > 0 and abs(angular_velocity) < min_angular_velocity:
                angular_velocity = np.sign(angular_velocity) * min_angular_velocity
            
            self.twist.linear.x = 0.0  
            self.twist.angular.z = angular_velocity
            self.vel_publisher.publish(self.twist)
            
            self.last_error = angle_error
            self.last_error_time = current_time
    

    def listener_callback(self, msg):
        """强化无目标时的状态重置"""
        if not self.vision_active:
            return
        max_confidence = 0.0
        best_roi = None
        # 只处理任务二且处于识别阶段的情况
        if self.star_yolov or self.task3_staryolov or (self.task_on2 and self.task2_step >= 2):
            for target in msg.targets:
                for roi in target.rois:
                    # 任务二只关注ap1/pe1
                    if self.task_on2 and self.task2_step >= 2:
                        if roi.type not in self.task2_priority_targets:
                            continue  # 过滤非目标
                    if roi.confidence > max_confidence and roi.confidence >= self.min_confidence:
                        max_confidence = roi.confidence
                        best_roi = roi
            
            # 无有效目标时重置状态
            if not best_roi:
                self.yolov_name = ""
                self.belive = 0.0
                self.task2_yolo_detected = False  # 明确标记无目标
                return

            # 有目标时正常更新
            self.yolov_x = best_roi.rect.x_offset + best_roi.rect.width / 2
            self.yolov_y = best_roi.rect.y_offset + best_roi.rect.height / 2
            self.yolov_name = best_roi.type.strip()
            self.belive = max_confidence
            self.trage_fand = True
            if self.task_on2 and self.task2_step >= 2:
                if self.yolov_name in self.task2_priority_targets:
                    self.task2_yolo_detected = True
            # 音频播报逻辑
            item_name_map = {
                "ch1": "辣椒",
                "to1": "西红柿",
                "on1": "洋葱",
                "pu1": "南瓜",
                "bad": "坏果",
                "ch2": "生辣椒",
                "to2": "生西红柿",
                "on2": "生洋葱",
                "pu2": "生南瓜",
                "ap1": "苹果",
                "ap2": "苹果",
                "pe1": "梨子",
                "pe2": "梨子"
            }
            if self.yolov_name in item_name_map:
                item_name = item_name_map[self.yolov_name]
                current_time = time.time()
                if (item_name != self.last_played_item or 
                    current_time - self.last_played_time > self.played_item_cooldown):
                    if item_name in self.audio_config["audio_paths"]:
                        self.play_local_wav(item_name)
                        self.last_played_item = item_name
                        self.last_played_time = current_time
        else:
            self.yolov_name = ""
            self.belive = 0.0

    def imu_callback(self, msg):
        quat = msg.orientation
        orientation_list = [quat.w, quat.x, quat.y, quat.z]
        roll, pitch, yaw = tfe.quat2euler(orientation_list, axes='sxyz')

        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        self.imu_yaw_history.append(yaw)
        self.imu_available = True

        if not self.calibration_done:
            if len(self.imu_yaw_history) == self.imu_yaw_history.maxlen:
                if self.calibration_start_time is None:
                    self.calibration_start_time = self.get_clock().now().seconds_nanoseconds()[0]
                current_time = self.get_clock().now().seconds_nanoseconds()[0]
                elapsed = current_time - self.calibration_start_time

                if elapsed >= self.CALIBRATION_DURATION:
                    avg_yaw = sum(self.imu_yaw_history) / len(self.imu_yaw_history)
                    self.imu_zero_offset = avg_yaw
                    self.calibration_done = True
                    self.get_logger().info(f"IMU校准完成: offset = {math.degrees(avg_yaw):+.2f}°")
        else:
            self.current_imu_yaw = yaw - self.imu_zero_offset
            self.current_imu_yaw = math.atan2(math.sin(self.current_imu_yaw), math.cos(self.current_imu_yaw))

        self.yaw_deg = math.degrees(self.current_imu_yaw)
        if self.goal_yaw is None:
            self.goal_yaw = self.yaw_deg
        self.imu_received = True

    def image_callback(self, msg):
        image_data = msg.data
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            self.get_logger().error("无法解码图像")
            self.qr_recognition_count = 0
            return
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        decoded_objects = pyzbar_decode(gray, symbols=[ZBarSymbol.QRCODE])
        current_qr_data = None
        
        if decoded_objects:
            for obj in decoded_objects:
                data = obj.data.decode('utf-8')
                if data:
                    current_qr_data = data
                    points = obj.polygon
                    if len(points) == 4:
                        pts = [(point.x, point.y) for point in points]
                        pts = np.array(pts, dtype=np.int32)
                        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
            if current_qr_data == self.qr_content_backup:
                self.qr_recognition_count += 1
                if self.qr_recognition_count >= 3 and not self.qr_processed:
                    if self.task_on2 and not self.task2_qr_scanned:
                        if self.task2_process_qr(current_qr_data):
                            self.qr_processed = True
                            self.last_qr_data = current_qr_data
                    else:
                        self.chinese_items, self.numbers = self.process_qr_data(current_qr_data)
                        self.get_logger().info(f"二维码稳定识别成功: {current_qr_data}")
                        self.get_logger().info(f"解析结果 - 物品: {self.chinese_items}, 编号: {self.numbers}")
                        
                        self.play_qr_content()
                        
                        self.qr_processed = True
                        self.last_qr_data = current_qr_data
                        
                        if self.task_on2:
                            self.task2_star_grap = True
                        if self.task3_getqr is False:
                            self.right_no()
                            self.task3_getqr = True
                            self.get_logger().info("已获取QR码信息")
            else:
                self.qr_content_backup = current_qr_data
                self.qr_recognition_count = 1
        else:
            self.qr_recognition_count = 0
            self.qr_content_backup = ""

    def play_qr_content(self):
        if not self.chinese_items or not self.numbers:
            self.get_logger().warn("二维码内容不完整，无法播报")
            return
            
        min_length = min(len(self.chinese_items), len(self.numbers))
        for i in range(min_length):
            item_name = self.chinese_items[i]
            item_num = self.numbers[i]
            
            if item_name in self.audio_config["audio_paths"]:
                self.play_local_wav(item_name)
                time.sleep(1)
            
            num_str = str(item_num)
            if num_str in self.audio_config["audio_paths"]:
                self.play_local_wav(num_str)
                time.sleep(1)
        self.vision_active = False
        self.get_logger().info("二维码识别与播报完成，已关闭视觉识别")

    def task1(self):
        if self.task_on1 and self.fand_num == 0 and not self.team_name_played:
            self.play_local_wav("队名")
            self.team_name_played = True
            time.sleep(1)
        
        if self.task_on1:
            if self.in_mid:
                if self.yolov_name == "ch1":
                    self.ch1_num += 1
                elif self.yolov_name == "to1":
                    self.to1_num += 1
                elif self.yolov_name == "pu1":
                    self.pu1_num += 1
                elif self.yolov_name == "on1":
                    self.on1_num += 1

                if self.yolov_name in self.bad_names:
                    self.get_logger().info(f"抓取坏果：{self.yolov_name}")
                    if self.fand_num in self.left_fans:
                        self.bad_grad()
                        self.fand_num += 1
                        self.in_mid = False
                        self.right_prep()
                    elif self.fand_num in self.right_fans:
                        self.bad_grad()
                        self.fand_num += 1
                        self.target += 0.5
                        self.in_mid = False
                        self.star_yolov = False
                        self.trage_fand = False
                        self.star_move = True
                elif self.yolov_name in self.ripe_names and self.fand_num in self.left_fans:
                        self.left_grad()
                        self.fand_num += 1
                        self.in_mid = False
                        self.right_prep()
                elif self.yolov_name in self.raw_names and self.fand_num in self.left_fans:
                    self.left_no()
                    self.fand_num += 1
                    self.in_mid = False
                    self.right_prep()
                elif self.yolov_name in self.ripe_names and self.fand_num in self.right_fans:
                    self.right_grad()
                    self.fand_num += 1
                    self.target += 0.5
                    self.in_mid = False
                    self.star_yolov = False
                    self.trage_fand = False
                    self.star_move = True
                elif self.yolov_name in self.raw_names and self.fand_num in self.right_fans:
                    self.right_no()
                    self.fand_num += 1
                    self.target += 0.5
                    self.in_mid = False
                    self.star_yolov = False
                    self.trage_fand = False
                    self.star_move = True
                

                self.yolov_name = ""
                self.yolov_x = 0.0
                self.yolov_y = 0.0

            if self.target_finsh:
                self.stop_robot()
                self.left_prep()
                self.star_yolov = True
                self.star_move = False
                self.target_finsh = False

            if self.fand_num >= 8:
                self.task_on1 = False
                self.gochang1 = True
                self.star_move = True
                self.stop_robot()
                self.get_logger().info("任务1完成，前往交换区")

    def go_chang1(self):
        if self.gochang1:
            self.get_logger().info(f"交换区移动步骤 {self.gochang1_step}/4")
            if self.gochang1_step == 1:
                self.target = 0.15
            elif self.gochang1_step == 2:
                self.target = 2.85
            elif self.gochang1_step == 3:
                self.target = 0.15

            if self.target_finsh:
                self.stop_robot()
                self.gochang1_step += 1
                if self.gochang1_step == 2:
                    self.is_turning = True
                    self.star_jigang = False
                    self.set_new_goal(-89.4)
                elif self.gochang1_step == 3:
                    self.is_turning = True
                    self.star_jigang = False
                    self.set_new_goal(89.4)
                elif self.gochang1_step == 4:
                    self.is_turning = False
                    self.star_jigang = True
                    self.qr_poss()
                    self.gochang1 = False
                    self.task_on3 = True
                    self.get_logger().info("开始执行任务3")
                self.target_finsh = False

    def task3(self):
        if self.task_on3:
            total_targets = len(self.numbers) if self.numbers else 0
            self.get_logger().info(f"任务3处理第 {self.task3_step+1}/{total_targets} 个目标")
            
            if self.task3_getqr:
                if len(self.numbers) > self.task3_step:
                    self.now_num = self.numbers[self.task3_step]
                    self.now_name = self.chinese_items[self.task3_step]
                    
                    self.get_logger().info(f"当前目标: {self.now_name}({self.now_num})")
                    
                

                    if not self.task3_goline:
                        if self.now_num in self.in3_left:
                            if self.task3_place == 1:
                                self.task3_goline = True
                            else:
                                self.right_to_left()
                        elif self.now_num in self.in3_mid:
                            self.task3_goline = True
                        elif self.now_num in self.in3_right:
                            if self.task3_place == 2:
                                self.task3_goline = True
                            else:
                                self.left_to_right()

                    if self.task3_goline and not self.task3_goon:
                        if self.now_num in self.on1:
                            self.target = 0.45
                        elif self.now_num in self.on2:
                            self.target = 0.87
                        elif self.now_num in self.on3:
                            self.target = 1.4
                        elif self.now_num in self.on4:
                            self.target = 1.87

                        self.go_target()
                        if self.target_finsh:
                            self.task3_goon = True
                            self.target_finsh = False
                            self.star_yolov = True
                            self.get_logger().info("到达目标位置，开始YOLO检测")
                    
                    if self.task3_goon and not self.task3_staryolov:
                        self.star_move = False
                        if self.now_num in self.in3_left:
                            self.left_prep()
                            self.arm_left = True
                            self.arm_right = False
                        elif self.now_num in self.in3_mid:
                            if self.task3_place == 2:
                                self.left_prep()
                                self.arm_left = True
                                self.arm_right = False
                            else:
                                self.right_prep()
                                self.arm_right = True
                                self.arm_left = False
                        elif self.now_num in self.in3_right:
                            self.right_prep()
                            self.arm_right = True
                            self.arm_left = False
                        self.task3_staryolov = True
                        self.check_and_grab()
                
                else:
                    self.task_on3 = False
                    self.task_all_done = True
                    self.gochang2_active = True
                    self.get_logger().info("任务3完成，启动最终任务")

    def check_and_grab(self):
        if not self.yolov_name:
            self.get_logger().info("等待YOLO检测结果...")
            return
            
        self.get_logger().info(f"检测到: {self.yolov_name}, 与目标 {self.now_name} 比对中...")
        
        name_mapping = {
            "辣椒": ["ch1", "ch2"],
            "西红柿": ["to1", "to2"],
            "洋葱": ["on1", "on2"],
            "南瓜": ["pu1", "pu2"],
            "坏果": ["bad"]
        }
        
        raw_fruits = ["ch2", "to2", "on2", "pu2"]
        
        if self.now_name not in name_mapping:
            self.get_logger().warn(f"未知目标名称: {self.now_name}, 不执行抓取")
            self.finish_current_target()
            return
            
        detected_name = self.yolov_name
        target_names = name_mapping[self.now_name]
        
        if detected_name not in target_names:
            self.get_logger().info(f"检测结果与目标不匹配: {detected_name} != {self.now_name}, 不执行抓取")
            self.finish_current_target()
            return
            
        if (detected_name in ["ch1", "to1", "on1", "pu1"] or 
            detected_name == "bad"):
            self.get_logger().info(f"符合抓取条件，执行抓取动作: {detected_name}")
            if self.arm_left:
                self.left_grad()
            elif self.arm_right:
                self.right_grad()
        elif detected_name in raw_fruits:
            self.get_logger().info(f"检测到生果 {detected_name}, 不执行抓取")
            if self.arm_left:
                self.left_no()
            elif self.arm_right:
                self.right_no()
        else:
            self.get_logger().info(f"未知类型 {detected_name}, 不执行抓取")
            if self.arm_left:
                self.left_no()
            elif self.arm_right:
                self.right_no()
        
        self.finish_current_target()

    def finish_current_target(self):
        self.task3_staryolov = False
        self.task3_goon = False
        self.task3_goline = False
        self.yolov_name = ""
        self.yolov_x = 0.0
        self.yolov_y = 0.0
        self.star_yolov = False
        self.task3_step += 1
        self.star_move = True
        self.trage_fand = False
        self.get_logger().info(f"当前目标处理完成，准备处理第 {self.task3_step+1} 个目标")

    def left_to_right(self):
        self.get_logger().info(f"左侧到右侧移动步骤 {self.l_rnum+1}/4")
        
        if self.lidar_data is not None:
            self.get_logger().info(f"激光测距: 当前距离={self.lidar_data:.3f}米, 目标距离={self.target:.3f}米")
        else:
            self.get_logger().warn("激光雷达数据不可用")
        
        if self.l_rnum == 0:
            self.target = 0.18
        elif self.l_rnum == 1:
            self.target = 2.9  # 2.93
        elif self.l_rnum == 2:
            self.target = 0.15  # 0.1
        elif self.l_rnum == 3:
            self.task3_place = 2
            self.task3_goline = True
            self.l_rnum = 0
            self.target_finsh = False
            self.task3_goline_num = 0
            self.get_logger().info("左侧到右侧移动完成")
            return

        if self.target_finsh and self.task3_goline_num > 5:
            self.stop_robot()
            self.l_rnum += 1
            self.get_logger().info(f"完成步骤{self.l_rnum}，进入步骤{self.l_rnum+1}")
            
            if self.l_rnum == 1:
                self.is_turning = True
                self.set_new_goal(-89.4)
                self.get_logger().info("开始左转90度")
            elif self.l_rnum == 2:
                self.is_turning = True
                self.set_new_goal(89.4)
                self.get_logger().info("开始右转90度")
            elif self.l_rnum == 3:
                self.is_turning = False
                self.get_logger().info("转弯完成")
            
            self.target_finsh = False
        else:
            if not self.target_finsh:
                self.get_logger().info("尚未到达目标距离")
            if self.task3_goline_num <= 5:
                self.get_logger().info(f"task3_goline_num={self.task3_goline_num} <= 5")
        
        self.task3_goline_num += 1

    def right_to_left(self):
        self.get_logger().info(f"右侧到左侧移动步骤 {self.r_lnum+1}/4")
        if self.r_lnum == 0:
            self.target = 0.18  # 0.1
        elif self.r_lnum == 1:
            self.target = 2.3
        elif self.r_lnum == 2:
            self.target = 0.15  # 0.1
        elif self.r_lnum == 4:
            self.task3_place = 1
            self.task3_goline = True
            self.r_lnum = 0
            self.target_finsh = False
            self.task3_goline_num = 0
            return

        if self.target_finsh and self.task3_goline_num > 5:
            self.stop_robot()
            self.r_lnum += 1
            if self.r_lnum == 1:
                self.is_turning = True
                self.set_new_goal(-89.4)
            elif self.r_lnum == 2:
                self.is_turning = True
                self.set_new_goal(89.4)
            elif self.r_lnum == 3:
                self.is_turning = False
            self.target_finsh = False
        self.task3_goline_num += 1
    
    
    def publish_goal(self):
        self.car_adjust()  
        
        if self.is_turning and self.turn_target_reached:
            self.is_turning = False
            self.turn_target_reached = False
            self.get_logger().debug("旋转状态已重置，恢复直线任务执行")
            return
        
        if not self.is_turning and self.calibration_done:
            if self.task_on3 and self.trage_fand and self.in_mid:
                self.gage_name()
            
            if self.task_on2:
                self.task2_main_logic()
                return
                
            if self.task3_goon and not self.task3_staryolov:
                if not hasattr(self, 'task3_goon_start_time'):
                    self.task3_goon_start_time = time.time()
                if time.time() - self.task3_goon_start_time > 5.0:
                    self.get_logger().warn("任务3目标移动超时，重置状态")
                    self.task3_goon = False
                    self.task3_goon_start_time = None
                    self.star_move = True
            
            self.go_target()
            self.task1()
            self.go_chang1()
            self.task3()
            self.grad_adiust()
            self.gochang2()

def main(args=None):
    rclpy.init(args=args)
    node = ImuAngleSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()
        node.stop_audio()
        node.get_logger().info("用户中断，程序退出")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()