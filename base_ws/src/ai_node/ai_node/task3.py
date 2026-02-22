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
            self.get_logger().info("串口连接成功：/dev/ttyUSB0")
        except serial.SerialException as e:
            self.get_logger().error(f"串口连接失败：{str(e)}，舵机控制功能将失效")
            self.serialHandle = None
        
        # 运动参数
        self.first_move = True
        self.base_linear_velocity = 4.2
        self.slow_linear_velocity = 4.2
        self.deceleration_threshold = 0.7
        self.align_speed = 0.04
        
        # 任务状态标记
        self.task_on1 = True
        self.task_on2 = False  # 任务二激活标记
        self.task2_qr_pose_adjusted = False  # 二维码姿态是否已调整
        self.task_on3 = False
        self.gochang3 =False
        self.gochang1 = False
        self.gochang2_active = False
        self.left_raw = False
        self.star_move = True
        self.team_name_played = False
        self.last_qr_data = None
        # 新增：二维码处理状态标记
        self.qr_processed = False
        self.qr_content_backup = ""  # 备份已识别的二维码内容
        self.qr_recognition_count = 0  # 连续识别计数，用于防抖

        # -------------------------- 任务2新增核心参数 --------------------------
        self.task2_step = 1  # 1:扫描二维码 2:处理风扇位 3:完成
        self.task2_fan_order = [1, 5, 2, 6, 3, 7, 4, 8]  # 处理顺序
        self.task2_left_dist = [2.3 - i*0.5 for i in range(4)]
        self.task2_right_dist = [2.2 - i*0.5 for i in range(4)]
        self.task2_fan_index = 0  # 当前处理索引
        self.task2_view_pos = 0  # 0:上方 1:下方 视角
        self.task2_qr_lines = []  # 二维码内容
        self.task2_qr_scanned = False  # 二维码是否扫描完成
        self.task2_dist_tol = 0.08  # 距离误差容忍度（提高精度）
        self.task2_yolo_detected = False  # 检测标记
        self.task2_fan_done = [False]*8  # 完成标记
        
        # 新增：抓取控制参数
        self.task2_grab_count = [0]*8  # 每棵树抓取次数
        self.task2_max_grabs = 2  # 单树最大抓取次数
        self.task2_current_fan_pos = None  # 当前处理的树
        self.task2_returning = False  # 是否在回位状态
        self.task2_view_timer = 0.0  # 视角识别计时器
        self.task2_recognition_timeout = 5.0  # 单视角识别超时
        self.task2_max_retry = 1  # 单视角重试次数
        self.task2_current_retry = 0  # 当前重试次数
        self.task2_priority_targets = {"ap1", "pe1"}  # 优先抓取目标
        # ----------------------------------------------------------------------

        # 任务2原有基础参数
        self.task2_fandnum = 0
        self.task2_star = False
        self.task2_go_qr = True
        self.task2_star_grap = False

        # 任务3参数
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

        # 最终阶段参数
        self.task_all_done = False
        self.go_to_9_step = 0
        self.gochang2_step = 0
        self.gochang2_turn_finished = False
        self.reached_2_5m = False

        # IMU参数
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

        # 转弯状态控制变量
        self.is_turning = False  # 是否正在执行转弯动作
        self.turn_target_reached = False  # 转弯目标是否达成
        
        # 直线校准禁用标记
        self.enable_straight_calibration = False  # 强制关闭直线校准

        # 交换区参数
        self.target = 0.75
        self.target_err = 0.1
        self.target_finsh = False
        self.gochang1_step = 1
        self.gochang3_step = 1

        # YOLO识别参数
        self.yolov_x = 0.0      
        self.yolov_y = 0.0
        self.yolov_name = ""
        self.belive = 0.0
        self.min_confidence = 0.65
        self.star_yolov = False
        self.vision_active = False

        # 任务1参数
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

        # 本地WAV音频配置
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
                "苹果树": "/root/audio_workspace/wav_files/apple_tree.wav",
                "梨子树": "/root/audio_workspace/wav_files/pear_tree.wav",
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
            }
        }
        self.audio_process = None  # 音频进程管理
        
        # 音频播放控制
        self.last_played_item = None
        self.played_item_cooldown = 3.0
        self.last_played_time = 0.0

        self.task3_yolo_start_time = None  # YOLO识别开始时间
        self.TASK3_YOLO_TIMEOUT = 5.0  # 任务3识别超时时间（8秒）
        self.task3_yolo_timeout_count = 0  # 超时次数计数

        # ROS订阅发布器
        self.imu_subscription = self.create_subscription(Imu, 'imu/data_raw', self.imu_callback, 10)
        self.yolo_subscription = self.create_subscription(PerceptionTargets, '/hobot_dnn_detection', self.listener_callback, 2)
        self.image_subscription = self.create_subscription(CompressedImage, 'image', self.image_callback, 10)
        self.lidar_subscription = self.create_subscription(Range, '/laser', self.lidar_callback, 10)
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_goal)
        self.twist = Twist()

        self.get_logger().info("机器人节点初始化完成，等待IMU校准...")

    # -------------------------- 任务2新增核心方法 --------------------------
    def task2_process_qr(self, data):
        """处理任务2的8行二维码，提取1-8位的苹果/梨子目标，仅播报果实序列"""
        lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
        
        # 1. 格式错误：行数不为8行
        if len(lines) != 8:
            self.get_logger().warn(f"任务2二维码格式错误！需8行，实际{len(lines)}行")
            time.sleep(1.5)
            return False
        
        # 2. 内容错误：包含非苹果/梨子的内容
        valid_fruits = {"苹果", "梨子"}
        invalid_lines = [line for line in lines if line not in valid_fruits]
        if invalid_lines:
            self.get_logger().warn(f"二维码包含无效内容：{invalid_lines}，仅支持苹果/梨子")
            time.sleep(1.5)
            return False
        
        # 3. 识别成功：保存数据并仅播报果实序列
        self.task2_qr_lines = lines
        self.task2_qr_scanned = True
        self.get_logger().info(f"任务2二维码扫描完成，目标序列：{self.task2_qr_lines}")
        
        # 播报1：提示“扫描完成”（可选）
        # self.play_local_wav("二维码扫描完成")
        time.sleep(2)  # 等待状态播报结束（若有）
        
        # 播报2：仅逐行播报果实（不播报位置编号）
        for fruit in self.task2_qr_lines:
            self.play_local_wav(fruit)  # 直接播报苹果/梨子
            time.sleep(1.2)  # 间隔1.2秒，避免播报重叠（可根据音频时长调整）
        
        return True

    def task2_get_target_dist(self, fan_pos):
        """根据风扇位（1-8）获取目标距离"""
        if 1 <= fan_pos <= 4:  # 左侧风扇位
            return self.task2_left_dist[fan_pos - 1]
        elif 5 <= fan_pos <= 8:  # 右侧风扇位
            return self.task2_right_dist[fan_pos - 5]
        else:
            self.get_logger().error(f"无效风扇位：{fan_pos}，仅支持1-8")
            return None

    def task2_move_to_fan(self, fan_pos,is_return=False):
        """移动到指定风扇位的理论位置（支持回位模式）"""
        target_dist = self.task2_get_target_dist(fan_pos)
        if target_dist is None:
            return False

        if self.lidar_data is None:
            self.get_logger().warn("无激光雷达数据，无法移动到目标风扇位")
            return False
        
        # 回位模式使用更高精度
        dist_tol = 0.05 if is_return else self.task2_dist_tol
        err = self.lidar_data - target_dist
        
        if abs(err) < dist_tol:
            self.stop_robot()
            self.get_logger().info(f"{'回位到' if is_return else '到达'}{fan_pos}号树，目标{target_dist}m，实际{self.lidar_data:.2f}m")
            return True
        else:
            speed = 0.5 * self.base_linear_velocity if is_return else self.base_linear_velocity
            if err > dist_tol:
                self.twist.linear.x = -speed  # 后退
            elif err < -dist_tol:
                self.twist.linear.x = speed   # 前进
            self.vel_publisher.publish(self.twist)
            return False
    
    def task2_return_to_fan(self):
        """返回当前风扇位的理论位置"""
        if self.task2_current_fan_pos is None:
            return False
        return self.task2_move_to_fan(self.task2_current_fan_pos, is_return=True)

    def task2_switch_view(self, view_pos):
        """切换识别视角（0：上方 1：下方）"""
        self.stop_robot()
        self.vision_active = True
        self.star_yolov = True
        self.task2_yolo_detected = False
        self.yolov_name = ""  # 清空识别结果
        self.in_mid = False   # 重置居中状态
        self.task2_view_pos = view_pos
        current_view = "上方" if view_pos == 0 else "下方"
        
        # 根据当前树位置切换机械臂姿态
        if view_pos == 0:
            if 1 <= self.task2_current_fan_pos <= 4:
                self.task2_left_prep_on()
            else:
                self.task2_right_prep_on()
        else:
            if 1 <= self.task2_current_fan_pos <= 4:
                self.task2_left_prep_under()
            else:
                self.task2_right_prep_under()
        
        self.get_logger().info(f"{self.task2_current_fan_pos}号树 - 切换到{current_view}视角（优先识别ap1/pe1）")
        time.sleep(1.5)  # 等待机械臂到位
    def task2_check_fruit(self):
        """检查是否识别到优先目标（ap1/pe1）并匹配目标类型"""
        if not self.yolov_name:
            return False
        
        # 只处理优先级目标
        if self.yolov_name not in self.task2_priority_targets:
            self.get_logger().info(f"忽略非优先目标{self.yolov_name}，只抓取ap1/pe1")
            return False
        
        # 检查抓取次数
        fan_idx = self.task2_current_fan_pos - 1
        if self.task2_grab_count[fan_idx] >= self.task2_max_grabs:
            self.get_logger().info(f"{self.task2_current_fan_pos}号树已达最大抓取次数")
            return False

        # 验证果实类型匹配
        target_fruit = self.task2_qr_lines[fan_idx]
        detected_type = "苹果" if self.yolov_name == "ap1" else "梨子"
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
        """任务2主逻辑：优化超时判断与非优先目标处理"""
        # 步骤1：扫描二维码（保持不变）
        if self.task2_step == 1:
            if not self.task2_qr_scanned:
                self.get_logger().info("任务2步骤1/3：扫描8行二维码中...")
                self.star_move = False
                if not self.task2_qr_pose_adjusted:
                    self.qr_poss()  # 调整摄像头姿态
                    self.task2_qr_pose_adjusted = True
            else:
                self.task2_qr_pose_adjusted = False
                self.task2_step = 2
                self.task2_fan_index = 0
                self.get_logger().info("进入风扇位处理阶段，按顺序1→5→2→6→3→7→4→8处理")
                return

        # 步骤2：处理风扇位
        elif self.task2_step == 2:
            if not hasattr(self, 'task2_sub_step'):
                self.task2_sub_step = ""
            
            # 所有树处理完成，进入步骤3
            if self.task2_fan_index >= len(self.task2_fan_order):
                self.task2_step = 3
                return
            
            # 当前处理的树
            self.task2_current_fan_pos = self.task2_fan_order[self.task2_fan_index]
            fan_idx = self.task2_current_fan_pos - 1

            # 检查是否已达最大抓取次数
            if self.task2_grab_count[fan_idx] >= self.task2_max_grabs:
                self.get_logger().info(f"{self.task2_current_fan_pos}号树已达最大抓取次数，跳过")
                self.task2_fan_index += 1
                self.task2_sub_step = ""
                self.task2_current_fan_pos = None
                return

            # 子步骤A：移动到理论位置（保持不变）
            if self.task2_sub_step == "":
                self.get_logger().info(f"处理{self.task2_current_fan_pos}号树（移动到理论位置）")
                if not self.task2_move_to_fan(self.task2_current_fan_pos):
                    return
                # 初始视角为上方
                self.task2_view_pos = 0
                self.task2_current_retry = 0
                self.task2_switch_view(self.task2_view_pos)
                self.task2_view_timer = time.time()
                self.task2_sub_step = "view_check"
                return

            # 子步骤B：视角识别与抓取（核心修改）
            if self.task2_sub_step == "view_check":
                current_view = "上方" if self.task2_view_pos == 0 else "下方"
                
                # 【修改1：优先判断超时，避免被居中调整阻塞】
                elapsed = time.time() - self.task2_view_timer
                if elapsed > self.task2_recognition_timeout:
                    self.get_logger().info(
                        f"{self.task2_current_fan_pos}号树{current_view}视角超时未识别（{elapsed:.1f}s），"
                        f"重试次数：{self.task2_current_retry}/{self.task2_max_retry}"
                    )
                    self.vision_active = False
                    self.star_yolov = False

                    # 重试机制
                    if self.task2_current_retry < self.task2_max_retry:
                        self.task2_current_retry += 1
                        self.task2_switch_view(self.task2_view_pos)
                        self.task2_view_timer = time.time()
                        return

                    # 重试失败，切换视角或推进下一步
                    self.task2_current_retry = 0
                    if self.task2_view_pos == 0:
                        self.get_logger().info("上层未识别到目标，切换到下层视角")
                        self.task2_view_pos = 1
                        self.task2_switch_view(self.task2_view_pos)
                        self.task2_view_timer = time.time()
                    else:
                        self.get_logger().info(f"上下层均未识别到目标，结束{self.task2_current_fan_pos}号树处理")
                        self.task2_fan_index += 1
                        self.task2_sub_step = ""
                        self.task2_current_fan_pos = None
                    return

                # 【修改2：仅对优先目标（ap1/pe1）执行居中，ap2/pe2不居中】
                if not self.in_mid and self.yolov_name in self.task2_priority_targets:
                    self.grad_adiust()
                    return

                # 【修改3：处理非优先目标（ap2/pe2），不抓取但继续计时】
                if self.yolov_name in ["ap2", "pe2"]:
                    self.get_logger().info(f"识别到{self.yolov_name}（仅播报，不抓取），继续等待优先目标...")
                    # 重置居中状态，避免阻塞超时
                    self.in_mid = False
                    self.trage_fand = False
                    return

                # 检测到目标，执行抓取（仅ap1/pe1会走到这里）
                if self.task2_check_fruit():
                    self.get_logger().info(f"执行{current_view}视角抓取{self.yolov_name}")
                    if self.task2_view_pos == 0:
                        self.task2_on_grad()
                    else:
                        self.task2_under_grad()
                    
                    # 更新抓取次数
                    self.task2_grab_count[fan_idx] += 1
                    self.get_logger().info(f"{self.task2_current_fan_pos}号树抓取次数：{self.task2_grab_count[fan_idx]}/{self.task2_max_grabs}")
                    
                    # 进入回位阶段
                    self.task2_sub_step = "return_to_position"
                    self.task2_returning = True
                    return

            # 子步骤C：抓取后回位（保持不变）
            if self.task2_sub_step == "return_to_position":
                current_view = "上方" if self.task2_view_pos == 0 else "下方"
                
                if self.task2_return_to_fan():
                    self.task2_returning = False
                    self.get_logger().info(f"{self.task2_current_fan_pos}号树回位完成")
                    # 重新切换到当前视角，准备再次识别
                    self.task2_switch_view(self.task2_view_pos)
                    self.task2_view_timer = time.time()
                    self.in_mid = False
                    
                    # 检查是否已达最大抓取次数
                    if self.task2_grab_count[fan_idx] >= self.task2_max_grabs:
                        self.get_logger().info(f"{self.task2_current_fan_pos}号树已达最大抓取次数")
                        if self.task2_view_pos == 0:
                            # 切换到下层视角
                            self.task2_view_pos = 1
                            self.task2_switch_view(self.task2_view_pos)
                            self.task2_view_timer = time.time()
                            self.task2_sub_step = "view_check"
                        else:
                            # 上下层均完成，进入下一棵树
                            self.task2_fan_index += 1
                            self.task2_sub_step = ""
                            self.task2_current_fan_pos = None
                    else:
                        self.get_logger().info(f"准备再次识别{self.task2_current_fan_pos}号树的{current_view}视角")
                        self.task2_sub_step = "view_check"
                return

        # 步骤3：任务2完成（保持不变）
        if self.task2_step == 3:
            self.stop_robot()
            self.get_logger().info("任务2所有风扇位处理完成！")
            self.task2_sub_step = ""  # 清空子步骤标记
            self.task2_view_pos = 0   # 重置视角
            self.vision_active = False  # 关闭视觉识别
            self.star_yolov = False     # 关闭YOLO检测
            self.trage_fand = False     # 关闭目标对准
            self.task2_yolo_detected = False  # 重置检测标记
            # 清除机械臂姿态标记
            self.arm_left = False
            self.arm_right = False
            self.play_local_wav("完成")  # 需确保存在"完成"音频
            time.sleep(2)
            # 任务2完成后自动激活任务1（可根据实际流程调整）
            self.task_on2 = False
            self.gochang1 = True
            self.star_move = True
            self.stop_robot()
    # -------------------------- 原有方法保留（新增任务2调用） --------------------------
    # 音频控制方法
    def stop_audio(self):
        """停止当前音频播放"""
        if self.audio_process and self.audio_process.poll() is None:
            self.audio_process.terminate()
            try:
                self.audio_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.audio_process.kill()
            self.get_logger().info("已停止当前音频播放")
        self.audio_process = None

    def play_local_wav(self, audio_key):
        """播放指定的本地WAV音频（异步播放，不阻塞任务）"""
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

    # 机器人控制基础方法
    def stop_robot(self):
        """停止机器人运动"""
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.vel_publisher.publish(self.twist)
        time.sleep(0.1)

    def move_robot1(self):
        """正向移动（基础速度）"""
        self.twist.linear.x = self.linear_velocity
        self.twist.angular.z = 0.0
        self.vel_publisher.publish(self.twist)

    def back_robot1(self):
        """反向移动（基础速度）"""
        self.twist.linear.x = -self.linear_velocity
        self.twist.angular.z = 0.0
        self.vel_publisher.publish(self.twist)

    def move_robot2(self):
        """微调移动（低速）"""
        self.twist.linear.x = 0.05
        self.twist.angular.z = 0.0
        self.vel_publisher.publish(self.twist)

    def back_robot2(self):
        """微调后退（低速）"""
        self.twist.linear.x = -0.05
        self.twist.angular.z = 0.0
        self.vel_publisher.publish(self.twist)

    # 舵机控制
    def setPWMServoMoveByArray(self, servos, servos_count, time):
        """控制多个舵机移动，带互斥锁和参数校验"""
        if not self.serialHandle or not self.serialHandle.isOpen():
            self.get_logger().error("串口未连接，无法控制舵机")
            return
        
        # 校验servos数组长度是否与舵机数量匹配
        expected_length = servos_count * 2  # 每个舵机需要2个参数（ID+角度）
        if len(servos) != expected_length:
            self.get_logger().error(f"舵机参数错误：预期{expected_length}个元素，实际{len(servos)}个")
            return
        
        # 组装指令帧
        buf = bytearray(b'\x55\x55')  # 帧头
        buf.append(servos_count * 3 + 5)  # 数据长度
        buf.append(LOBOT_CMD_SERVO_MOVE)  # 指令类型
        
        servos_count = max(1, min(servos_count, 254))  # 限制舵机数量范围
        buf.append(servos_count)
        
        # 限制时间最小值为100ms（避免硬件保护）
        time = max(100, min(time, 30000))
        buf.extend(time.to_bytes(2, 'little'))  # 时间（小端序）
        
        # 添加每个舵机的ID和角度
        for i in range(servos_count):
            buf.append(servos[i * 2])  # 舵机ID
            pos = max(500, min(servos[i * 2 + 1], 2500))  # 限制角度范围
            buf.extend(pos.to_bytes(2, 'little'))  # 角度（小端序）

        try:
            # 使用互斥锁确保指令完整发送
            with self.serial_lock:
                self.serialHandle.flushInput()  # 清空接收缓冲区，避免干扰
                self.serialHandle.write(buf)    # 发送指令
                self.serialHandle.flush()       # 强制刷新输出缓冲区
            self.get_logger().debug(f"舵机指令发送成功（长度：{len(buf)}字节）")
        except Exception as e:
            self.get_logger().error(f"舵机指令发送失败：{str(e)}")

    # 机械臂动作（保留原有方法，任务2复用）
    def qr_poss(self):
        """调整摄像头姿态读取QR码"""
        servos1 = [1, 2045, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos1, 4, 1100)
        time.sleep(1.1)
        
        servos2 = [3, 2100, 4, 1200, 9, 2150 ,8, 1600]
        self.setPWMServoMoveByArray(servos2, 4, 1100)
        time.sleep(1.1)

        self.vision_active = True
        self.star_yolov = True

    def left_prep(self):
        """左侧抓取准备"""
        servos1 = [1, 1075, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos1, 4, 1100)
        time.sleep(1.1)
        
        servos2 = [3, 1440, 4, 1930, 9, 1220]
        self.setPWMServoMoveByArray(servos2, 3, 1100)
        time.sleep(1.1)
        self.arm_left = True
        self.arm_right = False
        self.vision_active = True
        self.star_yolov = True

    def right_prep(self):
        """右侧抓取准备"""
        servos1 = [1, 2100, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos1, 4, 1100)
        time.sleep(1.1)
        
        servos2 = [3, 1440, 4, 1930, 9, 1220]
        self.setPWMServoMoveByArray(servos2, 3, 1100)
        time.sleep(1.1)
        self.arm_left = False
        self.arm_right = True

        self.vision_active = True
        self.star_yolov = True

    def left_grad(self):
        """左侧抓取动作"""
        servos = [3, 1080, 4, 1550, 9, 930]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.1)
        
        servos = [3, 1030, 4, 1235, 9, 1080]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.1)
        
        servos = [8, 2300]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(0.8)

        servos = [3,1600 , 4 , 600 ,9 ,2100 ,8 ,2000]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        servos = [1, 1600, 3,1600 , 4 , 600 ,9 ,2100]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.1)

        servos = [8 ,1600]
        self.setPWMServoMoveByArray(servos, 1, 800)  
        time.sleep(0.8)
        
        # 播放抓取的物品名称音频
        if self.now_name in self.audio_config["audio_paths"]:
            self.play_local_wav(self.now_name)
        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False

    def right_grad(self):
        """右侧抓取动作"""
        servos = [3, 1080, 4, 1550, 9, 930]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.1)
        
        servos = [3, 1030, 4, 1235, 9, 1080]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.1)
        
        servos = [8, 2300]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(0.8)

        servos = [3,1600 , 4 , 600 ,9 ,2100 ,8 ,2000]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        servos = [1, 1600, 3,1600 , 4 , 600 ,9 ,2100]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.1)

        servos = [8 ,1600]
        self.setPWMServoMoveByArray(servos, 1, 800)  
        time.sleep(0.8)
        
        # 播放抓取的物品名称音频
        if self.now_name in self.audio_config["audio_paths"]:
            self.play_local_wav(self.now_name)
        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False
    
    def bad_grad(self):
        """坏果抓取动作"""
        servos = [3, 1080, 4, 1550, 9, 930]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.1)
        
        servos = [3, 1030, 4, 1235, 9, 1045]
        self.setPWMServoMoveByArray(servos, 3, 1100)
        time.sleep(1.1)
        
        servos = [8, 2200]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(0.8)

        servos2 = [3, 1500, 4, 1725, 9, 1805]
        self.setPWMServoMoveByArray(servos2, 3, 1100)
        time.sleep(1.1)

        servos = [8 ,1600]
        self.setPWMServoMoveByArray(servos, 1, 800)  
        time.sleep(0.8)

        servos = [3,1700 , 4 , 600 ,9 ,2100 ,8 ,1800]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.1)

        servos = [1, 1600, 3,1700 , 4 , 600 ,9 ,2100]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.1)

        # 播放抓取的物品名称音频
        if self.now_name in self.audio_config["audio_paths"]:
            self.play_local_wav(self.now_name)
        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False

    def left_no(self):
        """左侧不抓取动作"""
        servos = [1, 1110, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos, 4, 1000)
        time.sleep(1)
        
        servos = [1, 1600]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(0.8)
        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False

    def right_no(self):
        """右侧不抓取动作"""
        servos = [1, 2100, 3, 1700, 4, 600, 9, 2100]
        self.setPWMServoMoveByArray(servos, 4, 1000)
        time.sleep(1)
        
        servos = [1, 1600]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(0.8)
        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False

    def task2_left_prep_on(self):
        """任务二左侧上方准备"""
        servos1 = [1, 1075, 3, 1700, 4, 600, 9, 1700, 8, 2100]
        self.setPWMServoMoveByArray(servos1, 5, 1100)
        time.sleep(1.1)
        
        servos2 = [3, 2140, 4, 1865, 9, 1725, 8, 2060]
        self.setPWMServoMoveByArray(servos2, 4, 1100)
        time.sleep(1.1)

        self.vision_active = True

    def task2_right_prep_on(self):
        """任务二右侧上方准备"""
        servos1 = [1, 2100, 3, 1700, 4, 600, 9, 1700, 8, 2100]
        self.setPWMServoMoveByArray(servos1, 5, 1100)
        time.sleep(1.5)
        
        servos2 = [3, 2140, 4, 1865, 9, 1725, 8, 2060]
        self.setPWMServoMoveByArray(servos2, 4, 1100)
        time.sleep(1.1)

        self.vision_active = True

    def task2_left_prep_under(self):
        """任务二左侧下方准备"""
        servos1 = [1, 1075, 3, 1700, 4, 600, 9, 2100, 8, 2100]
        self.setPWMServoMoveByArray(servos1, 5, 1100)
        time.sleep(1.5)
        
        servos2 = [3, 2140, 4, 1865, 9, 1935]
        self.setPWMServoMoveByArray(servos2, 3, 1100)
        time.sleep(1.5)

        self.vision_active = True

    def task2_right_prep_under(self):
        """任务二右侧下方准备"""
        servos1 = [1, 2100, 3, 1700, 4, 600, 9, 2100, 8, 2100]
        self.setPWMServoMoveByArray(servos1, 5, 1100)
        time.sleep(1.5)
        
        servos2 = [3, 2135, 4, 1880, 9, 1935]
        self.setPWMServoMoveByArray(servos2, 3, 1100)
        time.sleep(1.5)

        self.vision_active = True

    def task2_on_grad(self):
        """上侧抓取动作"""
        servos = [3, 1500, 4, 1540, 9, 995, 8, 1800]
        self.setPWMServoMoveByArray(servos, 4, 3000)
        time.sleep(3.5)

        servos = [3, 1460, 4, 1410, 9, 1060, 8, 1800]
        self.setPWMServoMoveByArray(servos, 4, 3000)
        time.sleep(3.5)
        
        servos = [8, 2300]
        self.setPWMServoMoveByArray(servos, 1, 1100)
        time.sleep(1.5)

        servos = [3,1800]
        self.setPWMServoMoveByArray(servos, 1, 500)  
        time.sleep(0.5)

        servos = [1, 1600, 3,1600 , 4 , 600 ,9 ,2300]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        servos = [8 ,1600]
        self.setPWMServoMoveByArray(servos, 1, 800)  
        time.sleep(1)

        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False

    def task2_under_grad(self):
        """下侧抓取动作"""
        servos = [3, 1145, 4, 1545, 9, 755, 8, 1800]
        self.setPWMServoMoveByArray(servos, 4, 3000)
        time.sleep(3.5)
        
        servos = [8, 2300]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(1)

        servos = [3, 1250, 4, 1820, 9, 755, 8, 2300]
        self.setPWMServoMoveByArray(servos, 4, 3000)
        time.sleep(3.5)

        servos = [3, 1800]
        self.setPWMServoMoveByArray(servos, 1, 800)
        time.sleep(1)

        servos = [3,1800 , 4 , 1200 ,9 ,2050 ,8 ,2100]
        self.setPWMServoMoveByArray(servos, 4, 2500)  
        time.sleep(1.5)

        

        servos = [1, 1600, 3,1600 , 4 , 600 ,9 ,2000]
        self.setPWMServoMoveByArray(servos, 4, 1100)  
        time.sleep(1.5)

        servos = [8 ,1600]
        self.setPWMServoMoveByArray(servos, 1, 800)  
        time.sleep(1)

        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False
    
    def pour(self):
        """倾倒动作"""
        servos = [11, 900]
        self.setPWMServoMoveByArray(servos, 1, 500)
        time.sleep(0.5)

        self.vision_active = False
        self.star_yolov = False
        self.trage_fand = False

    def last_servos(self):
        """回位动作"""
        servos = [1, 1600, 3, 1700, 4, 800, 9, 1800]
        self.setPWMServoMoveByArray(servos, 4, 1100)
        time.sleep(1.5)


    # 回调函数（修改image_callback以支持任务2二维码）
    def lidar_callback(self, msg):
        """激光雷达数据回调"""
        if self.star_jigang:
            self.lidar_data = msg.range
        else:
            self.lidar_data = None

    def process_qr_data(self, data):
        """处理QR码数据，优化对指定格式的解析"""
        # 处理空数据
        if not data.strip():
            return [], []
            
        # 分割行并过滤空行
        lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
        
        # 识别数字行（包含逗号分隔的数字）
        number_line_index = -1
        for i, line in enumerate(lines):
            if re.match(r'^[\d,]+$', line):  # 纯数字和逗号组成的行
                number_line_index = i
                break
                
        # 处理两种格式：带数字行和纯文字行
        if number_line_index != -1:
            # 文字部分为数字行之前的所有行
            chinese_part = lines[:number_line_index]
            # 数字部分为最后一个数字行
            numbers_part = lines[number_line_index]
        else:
            # 没有明确数字行，默认最后一行为数字
            if len(lines) >= 1:
                chinese_part = lines[:-1]
                numbers_part = lines[-1]
            else:
                chinese_part = lines
                numbers_part = ""
                
        # 提取文字项
        chinese_items = [item for item in chinese_part if item in ["洋葱", "南瓜", "西红柿", "辣椒"]]
        
        # 提取数字
        numbers = []
        if numbers_part:
            # 分割数字（支持逗号或空格分隔）
            number_strings = re.split(r'[, ]+', numbers_part)
            for num_str in number_strings:
                if num_str.isdigit():
                    numbers.append(int(num_str))
        
        return chinese_items, numbers

    def normalize_angle(self, angle):
        """角度归一化到[-180, 180)"""
        return (angle + 180) % 360 - 180

    def set_new_goal(self, delta_deg):
        """设置新的角度目标"""
        self.goal_yaw = self.normalize_angle(self.goal_yaw + delta_deg)
        self.is_goal_reached = False

    def go_target(self):
        """移动到目标距离（仅控制前进后退，不做方向校准）"""
        if self.star_move:
            if not self.lidar_data:
                self.get_logger().warn("无激光雷达数据")
                return
            
            err = self.lidar_data - self.target
            
            # 速度控制（仅前进后退，不涉及角度调整）
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
                # 仅控制直线速度，完全删除方向修正逻辑
                if err > self.target_err:
                    self.back_robot1()  # 只后退，不调整角度
                elif err < -self.target_err:
                    self.move_robot1()  # 只前进，不调整角度

    def gochang2(self):
        """调整流程：前进2.5米→停车→播报（保留后续任务运行能力）"""
        if self.gochang2_active:
            # 步骤0：初始化
            if self.gochang2_step == 0:
                self.get_logger().info("启动gochang2任务，初始化状态（前进2.5米→停车→播报）")
                # 运动状态标记
                self.reached_2_5m = False          # 是否到达目标：2.5米
                self.vehicle_stopped = False       # 小车是否最终停稳
                # 播报相关标记
                self.report_started = False
                self.report_complete = False
                # 移动参数（设置目标：2.5米）
                self.target = 2.5
                self.star_move = True              # 允许移动，执行前进
                self.get_logger().info("初始化完成，准备前进到2.5米")
                self.gochang2_step = 1  # 进入前进步骤

            # 步骤1：前进→到达2.5米目标点
            elif self.gochang2_step == 1:
                self.get_logger().info(f"gochang2步骤1/3：前往2.5米（当前距离：{self.lidar_data:.2f}米）")
                # 调用通用距离控制函数移动到2.5米
                self.go_target()
                # 到达2.5米后切换状态
                if self.target_finsh:
                    self.stop_robot()  # 到达后先停稳，避免惯性影响
                    time.sleep(1.5)
                    self.reached_2_5m = True
                    self.get_logger().info(f"已到达2.5米目标点")
                    self.gochang2_step = 2  # 进入停车步骤

            # 步骤2：确认停车状态
            elif self.gochang2_step == 2:
                self.get_logger().info("gochang2步骤2/3：确认停车状态")
                # 二次确认停车（双重保险）
                if not self.vehicle_stopped:
                    self.get_logger().info("强制停车，确保小车稳定")
                    self.stop_robot()
                    time.sleep(2.0)  # 延长等待时间，确保完全停稳
                    self.vehicle_stopped = True  # 标记小车已最终停稳
                
                # 执行倾倒动作（保留原有功能）
                # self.pour()
                self.gochang2_step = 3  # 进入播报步骤

            # 步骤3：执行播报（适配0ge格式）
            elif self.gochang2_step == 3:
                self.get_logger().info("gochang2步骤3/3：开始播报采收情况")
                # 仅执行一次播报
                if not self.report_started:
                    self.report_started = True
                    # 果实列表（固定顺序）
                    fruit_count_map = [
                        ("南瓜", self.pu1_num),
                        ("辣椒", self.ch1_num),
                        ("西红柿", self.to1_num),
                        ("洋葱", self.on1_num)
                    ]
                    # 逐个播报
                    for fruit_name, count in fruit_count_map:
                        # 播放果实名称
                        if fruit_name in self.audio_config["audio_paths"]:
                            self.play_local_wav(fruit_name)
                            time.sleep(1.2)
                        # 播放数量
                        num_audio_key = f"{count}ge"
                        if num_audio_key in self.audio_config["audio_paths"]:
                            self.play_local_wav(num_audio_key)
                            time.sleep(1)
                    
                    self.report_complete = True
                    self.get_logger().info("播报完成")
                    # 仅关闭当前任务标记，不锁定运动状态
                    self.gochang2_active = False
                    self.gochang3 = True
                    self.target = 0.0
                    self.target_finsh = False
                    self.star_move = True
                    self.gochang2_step = 0
                    self.get_logger().info("gochang2流程完成，准备运行后续任务")

    def gage_name(self):
        """判断果实是否符合抓取条件（修复生果强制抓取问题）"""
        if self.task_on3:
            # 判断果实是否匹配
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
            
            # 执行抓取或不抓取动作（核心修复：生果/不匹配时执行不抓取）
            if self.name_right and self.yolov_name:
                if self.arm_left:
                    self.left_grad()
                elif self.arm_right:
                    self.right_grad()
                # 更新计数
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
                self.get_logger().info(f"任务3抓取坏果：{self.yolov_name}（不计数）")
                if self.arm_left:
                    self.bad_grad()
                elif self.arm_right:
                    self.bad_grad()
            else:
                # 修复：生果/不匹配目标时执行“不抓取”动作
                self.get_logger().info(f"检测到生果或不匹配目标：{self.yolov_name}，不执行抓取")
                if self.arm_left:
                    self.left_no()
                elif self.arm_right:
                    self.right_no()
            
            # 重置状态
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

    def grad_adiust(self):
        """调整机械臂对准目标（仅在此函数内添加左右区分，避免姿态异常）"""
        if self.trage_fand:
            # 任务二：强制根据树位置判断左右（1-4为左，5-8为右），避免姿态异常
            if self.task_on2 and self.task2_step == 2 and self.task2_current_fan_pos is not None:
                # 手动设置左右标记（根据当前树位置，无需依赖外部状态）
                if 1 <= self.task2_current_fan_pos <= 4:
                    self.arm_left = True
                    self.arm_right = False
                else:
                    self.arm_left = False
                    self.arm_right = True
            # 其他任务：保留原有姿态判断（若未设置则默认左侧，避免异常）
            else:
                if not (self.arm_left ^ self.arm_right):
                    self.arm_left = True  # 默认左侧，避免姿态异常
                    self.arm_right = False

            # 任务二居中精度更高（±10像素）+ 区分左右
            if self.task_on2 and self.task2_step == 2:
                # 左侧机械臂（1-4号树）：目标偏左→前进，偏右→后退
                if self.arm_left:
                    if self.yolov_x < 310.0:
                        self.twist.linear.x = -self.align_speed  # 左偏→前进
                        self.in_mid = False
                    elif self.yolov_x > 330.0:
                        self.twist.linear.x = self.align_speed  # 右偏→后退
                        self.in_mid = False
                    else:
                        self.in_mid = True
                        self.stop_robot()
                        self.get_logger().info(f"左侧机械臂 - 目标居中完成（X={self.yolov_x:.1f}）")
                # 右侧机械臂（5-8号树）：目标偏左→后退，偏右→前进
                else:
                    if self.yolov_x < 310.0:
                        self.twist.linear.x = self.align_speed  # 左偏→后退
                        self.in_mid = False
                    elif self.yolov_x > 330.0:
                        self.twist.linear.x = -self.align_speed  # 右偏→前进
                        self.in_mid = False
                    else:
                        self.in_mid = True
                        self.stop_robot()
                        self.get_logger().info(f"右侧机械臂 - 目标居中完成（X={self.yolov_x:.1f}）")
                self.vel_publisher.publish(self.twist)

            # 其他任务保持原有逻辑+左右区分
            else:
                if self.arm_left:
                    # 左侧机械臂：左偏→前进，右偏→后退
                    if self.yolov_x < 230.0:
                        self.twist.linear.x = -self.align_speed
                        self.in_mid = False
                    elif self.yolov_x > 410.0:
                        self.twist.linear.x = self.align_speed
                        self.in_mid = False
                    else:
                        self.in_mid = True
                        self.stop_robot()
                else:
                    # 右侧机械臂：左偏→后退，右偏→前进
                    if self.yolov_x < 230.0:
                        self.twist.linear.x = self.align_speed
                        self.in_mid = False
                    elif self.yolov_x > 410.0:
                        self.twist.linear.x = -self.align_speed
                        self.in_mid = False
                    else:
                        self.in_mid = True
                        self.stop_robot()
                self.vel_publisher.publish(self.twist)

    def car_adjust(self):
        """仅在主动旋转时工作，直线运动时完全跳过（彻底关闭直线校准）"""
        # 未完成校准时停止机器人
        if not self.calibration_done:
            self.stop_robot()
            return

        # 无IMU数据时警告并停止
        if not self.imu_received:
            self.get_logger().warn("无IMU数据，无法进行角度调整")
            self.stop_robot()
            return

        # 仅旋转时执行调整
        if not self.is_turning:  # 非旋转状态（直线运动），直接返回
            return

        # 计算角度误差（目标角度与当前角度的差值）
        angle_error = self.normalize_angle(self.goal_yaw - self.yaw_deg)
        
        # 角度误差小于停止阈值时，停止转弯
        if abs(angle_error) < self.angle_threshold_stop:
            if not self.is_goal_reached:
                self.stop_robot()
                self.is_goal_reached = True
                self.star_jigang = True  
                self.turn_target_reached = True  # 标记转弯目标达成
                self.get_logger().info(f"转弯完成，当前角度: {self.yaw_deg:.2f}°，目标角度: {self.goal_yaw:.2f}°")
            return
        # 角度误差大于启动阈值时，激活转弯
        elif abs(angle_error) > self.angle_threshold_start:
            self.is_goal_reached = False

        # 执行转弯控制（仅处理转弯，不涉及直线调整）
        if not self.is_goal_reached:
            # 输出转弯状态日志
            if self.task_on1:
                self.get_logger().info(f"任务1转弯调整 - 误差: {angle_error:.2f}°")
            elif self.task_on3:
                self.get_logger().info(f"任务3转弯调整 - 误差: {angle_error:.2f}°")
            elif self.gochang2_active:
                self.get_logger().info(f"最终阶段转弯调整 - 误差: {angle_error:.2f}°")

            self.star_jigang = False  # 转弯时禁用激光雷达直线控制
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            # 计算误差导数（用于PD控制）
            if self.last_error_time is not None:
                dt = current_time - self.last_error_time
                error_derivative = (angle_error - self.last_error) / dt if dt > 0 else 0
            else:
                error_derivative = 0
            
            # 计算角速度（PD控制）
            angular_velocity = (
                self.kp_angular * angle_error + 
                self.kd_angular * error_derivative
            )
            
            # 限制角速度范围
            angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)
            
            # 小误差时减速，提高精度
            # if abs(angle_error) < self.slowdown_threshold:
            #     angular_velocity *= abs(angle_error) / self.slowdown_threshold
            # 小误差时减速（加入放大系数，解决12度阈值下微调过慢问题）
            if abs(angle_error) < self.slowdown_threshold:
                # scale_factor：放大系数（>1表示加快微调速度，建议从1.5开始测试）
                scale_factor = 0.05  
                # 原比例 × 放大系数，增大角速度，让微调更灵敏
                angular_velocity *= (abs(angle_error) / self.slowdown_threshold) * scale_factor
            # 确保最小角速度，避免卡滞
            min_angular_velocity = 0.1
            if abs(angular_velocity) > 0 and abs(angular_velocity) < min_angular_velocity:
                angular_velocity = np.sign(angular_velocity) * min_angular_velocity
            
            # 仅发布角速度（直线速度强制为0）
            self.twist.linear.x = 0.0  
            self.twist.angular.z = angular_velocity
            self.vel_publisher.publish(self.twist)
            
            # 记录误差和时间，用于下次导数计算
            self.last_error = angle_error
            self.last_error_time = current_time
    

    def listener_callback(self, msg):
        """YOLO识别结果回调（支持任务2的ap1/ap2/pe1/pe2检测）"""
        if not self.vision_active:
            return
        max_confidence = 0.0
        best_roi = None
        # 同时支持原有任务和任务2的YOLO检测
        if self.star_yolov or self.task3_staryolov or (self.task_on2 and self.task2_step >= 2):
            for target in msg.targets:
                for roi in target.rois:
                    if roi.confidence > max_confidence and roi.confidence >= self.min_confidence:
                        max_confidence = roi.confidence
                        best_roi = roi
            
            if best_roi:
                self.yolov_x = best_roi.rect.x_offset + best_roi.rect.width / 2
                self.yolov_y = best_roi.rect.y_offset + best_roi.rect.height / 2
                self.yolov_name = best_roi.type.strip()
                self.belive = max_confidence
                self.trage_fand = True
                # 任务2的果树检测标记
                if self.task_on2 and self.task2_step >= 2:
                    if self.yolov_name in ["ap1", "ap2", "pe1", "pe2"]:
                        self.task2_yolo_detected = True
                
                if self.task_on3 and self.task3_staryolov:
                    self.task3_yolo_start_time = None
                # 播放识别到的物品音频（带冷却控制）
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
                    "ap2": "生苹果",
                    "pe1": "梨子",
                    "pe2": "生梨子"
                }
                if self.yolov_name in item_name_map:
                    item_name = item_name_map[self.yolov_name]
                    current_time = time.time()
                    # 任务2专属判断：仅ap/pe系列触发播报
                    if self.task_on2 and self.task2_step >= 2:
                        if self.yolov_name.startswith(("ap", "pe")):
                            if (item_name != self.last_played_item or 
                                current_time - self.last_played_time > self.played_item_cooldown):
                                if item_name in self.audio_config["audio_paths"]:
                                    self.play_local_wav(item_name)
                                    self.last_played_item = item_name
                                    self.last_played_time = current_time
                    # 其他任务（任务1/3）保留原播报逻辑
                    else:
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
        """IMU数据回调与校准"""
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
        """图像回调（支持任务2的8行二维码扫描）"""
        image_data = msg.data
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            self.get_logger().error("无法解码图像")
            self.qr_recognition_count = 0  # 重置识别计数
            return
            
        # 图像预处理：增强对比度和降噪，提高二维码识别率
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯模糊降噪
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 自适应二值化
        
        # 尝试识别二维码
        decoded_objects = pyzbar_decode(gray, symbols=[ZBarSymbol.QRCODE])
        current_qr_data = None
        
        if decoded_objects:
            for obj in decoded_objects:
                data = obj.data.decode('utf-8')
                if data:
                    current_qr_data = data
                    # 绘制二维码边框
                    points = obj.polygon
                    if len(points) == 4:
                        pts = [(point.x, point.y) for point in points]
                        pts = np.array(pts, dtype=np.int32)
                        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
            # 连续识别防抖：连续3次识别到相同内容才确认有效
            if current_qr_data == self.qr_content_backup:
                self.qr_recognition_count += 1
                if self.qr_recognition_count >= 3 and not self.qr_processed:
                    # 任务2优先级：优先处理8行二维码
                    if self.task_on2 and not self.task2_qr_scanned:
                        if self.task2_process_qr(current_qr_data):
                            self.qr_processed = True
                            self.last_qr_data = current_qr_data
                    else:
                        # 原有任务的二维码处理
                        self.chinese_items, self.numbers = self.process_qr_data(current_qr_data)
                        self.get_logger().info(f"二维码稳定识别成功: {current_qr_data}")
                        self.get_logger().info(f"解析结果 - 物品: {self.chinese_items}, 编号: {self.numbers}")
                        
                        # 触发播报（仅播报一次）
                        self.play_qr_content()
                        
                        self.qr_processed = True
                        self.last_qr_data = current_qr_data
                        
                        # 任务2和任务3的触发逻辑
                        if self.task_on2:
                            self.task2_star_grap = True
                        if self.task3_getqr is False:
                            self.right_no()
                            self.task3_getqr = True
                            self.get_logger().info("已获取QR码信息")
            else:
                # 内容变化时重置计数
                self.qr_content_backup = current_qr_data
                self.qr_recognition_count = 1
        else:
            # 未识别到二维码时重置计数
            self.qr_recognition_count = 0
            self.qr_content_backup = ""

    def play_qr_content(self):
        """播报二维码中的物品名称和编号"""
        if not self.chinese_items or not self.numbers:
            self.get_logger().warn("二维码内容不完整，无法播报")
            return
            
        # 确保物品和编号数量一致
        min_length = min(len(self.chinese_items), len(self.numbers))
        for i in range(min_length):
            item_name = self.chinese_items[i]
            item_num = self.numbers[i]
            
            # 播放物品名称
            if item_name in self.audio_config["audio_paths"]:
                self.play_local_wav(item_name)
                time.sleep(1)  # 等待播放完成
            
            # 播放编号
            num_str = str(item_num)
            if num_str in self.audio_config["audio_paths"]:
                self.play_local_wav(num_str)
                time.sleep(1)  # 等待播放完成
        self.vision_active = False
        self.get_logger().info("二维码识别与播报完成，已关闭视觉识别")

    # 任务处理（保留原有方法）
    def task1(self):
        """处理任务1"""
        if self.task_on1 and self.fand_num == 0 and not self.team_name_played:
            self.play_local_wav("队名")
            self.team_name_played = True
            time.sleep(1)
        
        if self.task_on1:
            if self.in_mid:
                # 更新计数
                if self.yolov_name == "ch1":
                    self.ch1_num += 1
                elif self.yolov_name == "to1":
                    self.to1_num += 1
                elif self.yolov_name == "pu1":
                    self.pu1_num += 1
                elif self.yolov_name == "on1":
                    self.on1_num += 1

                # 执行抓取逻辑
                if self.yolov_name in self.bad_names:
                    self.get_logger().info(f"抓取坏果：{self.yolov_name}（不计数）")
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
                

                # 重置状态
                self.yolov_name = ""
                self.yolov_x = 0.0
                self.yolov_y = 0.0

            if self.target_finsh:
                self.stop_robot()
                self.left_prep()
                self.star_yolov = True
                self.star_move = False
                self.target_finsh = False

            # 任务1完成条件
            if self.fand_num >= 8:
                self.task_on1 = False
                self.gochang1 = True
                self.star_move = True
                self.stop_robot()
                self.get_logger().info("任务1完成，前往交换区")

    def go_chang1(self):
        """前往交换区（优化旋转时激光雷达禁用）"""
        if self.gochang1:
            self.get_logger().info(f"交换区移动步骤 {self.gochang1_step}/4")
            if self.gochang1_step == 1:
                self.target = 0.19
            elif self.gochang1_step == 2:
                self.target = 2.85
            elif self.gochang1_step == 3:
                self.target = 0.19

            if self.target_finsh:
                self.stop_robot()
                self.gochang1_step += 1
                if self.gochang1_step == 2:
                    # 准备转弯：启用旋转状态 + 禁用激光雷达
                    self.is_turning = True
                    self.star_jigang = False  # 旋转时禁用激光雷达直线控制
                    self.set_new_goal(-89.4)
                elif self.gochang1_step == 3:
                    # 准备转弯：启用旋转状态 + 禁用激光雷达
                    self.is_turning = True
                    self.star_jigang = False  # 旋转时禁用激光雷达直线控制
                    self.set_new_goal(89.4)
                elif self.gochang1_step == 4:
                    self.is_turning = False  # 关闭转弯状态
                    self.star_jigang = True   # 恢复激光雷达
                    self.qr_processed = False        # 允许重新识别二维码
                    self.qr_content_backup = ""       # 清空历史识别内容
                    self.qr_recognition_count = 0
                    self.qr_poss()
                    self.gochang1 = False
                    self.task_on3 = True
                    self.get_logger().info("开始执行任务3")
                self.target_finsh = False
    
    def go_chang3(self):
        """前往任务二"""
        if self.gochang3:
            self.get_logger().info(f"交换区移动步骤 {self.gochang3_step}/4")
            if self.gochang3_step == 1:
                self.target = 0.19
            elif self.gochang3_step == 2:  # 修复原代码拼写错误（gochang1step→gochang1_step）
                self.target = 1.4
            elif self.gochang3_step == 3:
                self.target = 2.55#2.6

            if self.target_finsh:
                self.stop_robot()
                self.gochang3_step += 1
                if self.gochang3_step == 2:
                    self.is_turning = True
                    self.star_jigang = False
                    self.set_new_goal(-89.4)
                elif self.gochang3_step == 3:
                    self.is_turning = True
                    self.star_jigang = False
                    self.set_new_goal(89.0)#-89.4
                elif self.gochang3_step == 4:
                    self.is_turning = False
                    self.star_jigang = True
                    self.qr_poss()
                    self.gochang3 = False
                    self.task_on2 = True  # 交换区完成后启动任务二
                    self.task_on3 = False  # 暂不启动任务三
                    self.get_logger().info("开始执行任务二")
                self.target_finsh = False

    def task3(self):
        """处理任务3"""
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
                            self.target = 0.46
                        elif self.now_num in self.on2:
                            self.target = 0.88
                        elif self.now_num in self.on3:
                            self.target = 1.4
                        elif self.now_num in self.on4:
                            self.target = 1.87

                        self.go_target()
                        if self.target_finsh:
                            self.task3_goon = True
                            self.target_finsh = False
                            # 到达目标位置后启动YOLO检测
                            self.star_yolov = True
                            self.task3_yolo_start_time = time.time()
                            self.get_logger().info("到达目标位置，开始YOLO检测")
                    
                    if self.task3_goon and not self.task3_staryolov:
                        self.star_move = False
                        # 明确设置机械臂姿态标记
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
                        # 准备完成后检查是否需要抓取
                        self.check_and_grab()
                
                else:
                    self.task_on3 = False
                    self.task_all_done = True
                    self.gochang2_active = True
                    self.get_logger().info("任务3完成，启动最终任务")

                if self.task3_staryolov and self.task3_yolo_start_time is not None:
                    elapsed = time.time() - self.task3_yolo_start_time
                    if elapsed >= self.TASK3_YOLO_TIMEOUT:
                        self.get_logger().warn(f"任务3 YOLO识别超时（{elapsed:.1f}秒），跳过当前目标")
                        # 执行不抓取动作
                        if self.arm_left:
                            self.left_no()
                        elif self.arm_right:
                            self.right_no()
                        # 记录超时次数
                        self.task3_yolo_timeout_count += 1
                        self.get_logger().info(f"任务3累计超时次数：{self.task3_yolo_timeout_count}")
                        # 重置状态，进入下一个目标
                        self.finish_current_target()
                        # 重置超时计时器
                        self.task3_yolo_start_time = None

    def check_and_grab(self):
        """检查果实是否符合抓取条件并执行相应动作（强化生果过滤）"""
        # 等待YOLO检测结果
        if not self.yolov_name:
            self.get_logger().info("等待YOLO检测结果...")
            return
            
        self.get_logger().info(f"检测到: {self.yolov_name}, 与目标 {self.now_name} 比对中...")
        
        # 映射二维码名称到YOLO识别结果
        name_mapping = {
            "辣椒": ["ch1", "ch2"],  # ch1:成熟, ch2:生
            "西红柿": ["to1", "to2"],
            "洋葱": ["on1", "on2"],
            "南瓜": ["pu1", "pu2"],
            "坏果": ["bad"]
        }
        
        # 明确生果类型列表
        raw_fruits = ["ch2", "to2", "on2", "pu2"]
        
        # 检查是否为已知名称
        if self.now_name not in name_mapping:
            self.get_logger().warn(f"未知目标名称: {self.now_name}, 不执行抓取")
            self.finish_current_target()
            return
            
        # 检查检测结果是否匹配目标
        detected_name = self.yolov_name
        target_names = name_mapping[self.now_name]
        
        if detected_name not in target_names:
            self.get_logger().info(f"检测结果与目标不匹配: {detected_name} != {self.now_name}, 不执行抓取")
            self.finish_current_target()
            return
            
        # 判断是否为成熟果实或坏果（核心修复：明确区分生果）
        if (detected_name in ["ch1", "to1", "on1", "pu1"] or  # 成熟果实
            detected_name == "bad"):  # 坏果
            self.get_logger().info(f"符合抓取条件，执行抓取动作: {detected_name}")
            if self.arm_left:
                self.left_grad()
            elif self.arm_right:
                self.right_grad()
        elif detected_name in raw_fruits:
            # 生果强制不抓取
            self.get_logger().info(f"检测到生果 {detected_name}, 不执行抓取")
            if self.arm_left:
                self.left_no()
            elif self.arm_right:
                self.right_no()
        else:
            # 其他不匹配目标不抓取
            self.get_logger().info(f"未知类型 {detected_name}, 不执行抓取")
            if self.arm_left:
                self.left_no()
            elif self.arm_right:
                self.right_no()
        
        # 完成当前目标处理
        self.finish_current_target()

    def finish_current_target(self):
        """完成当前目标处理，重置状态并准备下一个目标"""
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
        self.task3_yolo_start_time = None
        self.get_logger().info(f"当前目标处理完成，准备处理第 {self.task3_step+1} 个目标")

    def left_to_right(self):
        """从左侧移动到右侧区域"""
        self.get_logger().info(f"左侧到右侧移动步骤 {self.l_rnum+1}/4")
        
        # 添加详细的步骤状态日志
        self.get_logger().info(f"当前状态: target_finsh={self.target_finsh}, task3_goline_num={self.task3_goline_num}")
        
        if self.lidar_data is not None:
            self.get_logger().info(f"激光测距: 当前距离={self.lidar_data:.3f}米, 目标距离={self.target:.3f}米")
        else:
            self.get_logger().warn("激光雷达数据不可用")
        
        if self.l_rnum == 0:
            self.target = 0.19
        elif self.l_rnum == 1:
            self.target = 2.92  # 2.93
        elif self.l_rnum == 2:
            self.target = 0.19  # 0.1
        elif self.l_rnum == 3:  # 修正：应该是3而不是4
            self.task3_place = 2
            self.task3_goline = True
            self.l_rnum = 0
            self.target_finsh = False
            self.task3_goline_num = 0
            self.get_logger().info("左侧到右侧移动完成")
            return

        # 检查是否满足进入下一步的条件
        if self.target_finsh and self.task3_goline_num > 5:
            self.stop_robot()
            self.l_rnum += 1
            self.get_logger().info(f"完成步骤{self.l_rnum}，进入步骤{self.l_rnum+1}")
            
            if self.l_rnum == 1:
                self.is_turning = True  # 开始转弯
                self.set_new_goal(-89.4)
                self.get_logger().info("开始左转90度")
            elif self.l_rnum == 2:
                self.is_turning = True  # 开始转弯
                self.set_new_goal(90)
                self.get_logger().info("开始右转90度")
            elif self.l_rnum == 3:
                self.is_turning = False  # 结束转弯
                self.get_logger().info("转弯完成")
            
            self.target_finsh = False
        else:
            # 如果不满足条件，记录原因
            if not self.target_finsh:
                self.get_logger().info("尚未到达目标距离")
            if self.task3_goline_num <= 5:
                self.get_logger().info(f"task3_goline_num={self.task3_goline_num} <= 5")
        
        self.task3_goline_num += 1

    def right_to_left(self):
        """从右侧移动到左侧区域"""
        self.get_logger().info(f"右侧到左侧移动步骤 {self.r_lnum+1}/4")
        if self.r_lnum == 0:
            self.target = 0.19  # 0.1
        elif self.r_lnum == 1:
            self.target = 2.31
        elif self.r_lnum == 2:
            self.target = 0.19  # 0.1
        elif self.r_lnum == 4:  # 修复步骤判断错误
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
                self.is_turning = True  # 开始转弯
                self.set_new_goal(-89.4)
            elif self.r_lnum == 2:
                self.is_turning = True  # 开始转弯
                self.set_new_goal(89.4)
            elif self.r_lnum == 3:
                self.is_turning = False  # 结束转弯
            self.target_finsh = False
        self.task3_goline_num += 1
    
    
    def publish_goal(self):
        """主循环：发布速度指令并处理任务（新增任务二调用）"""
        # 1. 先执行旋转调整（仅在is_turning=True时生效）
        self.car_adjust()  
        
        # 2. 旋转完成后，重置旋转状态（避免阻塞后续直线任务）
        if self.is_turning and self.turn_target_reached:
            self.is_turning = False  # 重置旋转标记
            self.turn_target_reached = False  # 重置目标达成标记
            self.get_logger().debug("旋转状态已重置，恢复直线任务执行")
            return  # 等待下一周期再执行直线任务，确保旋转完全停止
        
        # 3. 非旋转状态且校准完成，执行直线任务逻辑
        if not self.is_turning and self.calibration_done:
            # 新增：任务三抓取触发（YOLO识别到目标且对准完成）
            if self.task_on3 and self.trage_fand and self.in_mid:
                self.gage_name()  # 触发抓取判断
            
            # 新增：任务二主逻辑调用（优先级低于旋转，高于其他任务）
            if self.task_on2:
                self.task2_main_logic()
                return  # 任务二执行时，暂不执行其他任务
                
            # 任务超时容错（保留原逻辑）
            if self.task3_goon and not self.task3_staryolov:
                if not hasattr(self, 'task3_goon_start_time'):
                    self.task3_goon_start_time = time.time()
                if time.time() - self.task3_goon_start_time > 5.0:
                    self.get_logger().warn("任务3目标移动超时，重置状态")
                    self.task3_goon = False
                    self.task3_goon_start_time = None
                    self.star_move = True
            
            # 执行各任务的直线运动（仅距离控制，无方向修正）
            self.go_target()  # 激光雷达距离控制（纯直线，不调方向）
            self.task1()
            self.go_chang1()
            self.go_chang3()
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