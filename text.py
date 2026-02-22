#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import serial
import time
import argparse
from typing import Dict, List, Tuple


class ServoController:
    """舵机控制器（平滑控制版，仅显示最终角度反馈）"""
    
    def __init__(self, port: str = '/dev/ttyUSB1', baudrate: int = 9600):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=0.5,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
        if not self.ser.is_open:
            raise Exception(f"无法打开串口: {port}")
        print(f"串口已打开: {port}")

        # 当前位置存储（key：舵机ID，value：位置值500-2500）
        self.current_pos: Dict[int, int] = {}

    def close(self):
        if self.ser.is_open:
            self.ser.close()
            print("串口已关闭")

    def _get_display_angle(self, servo_id: int, pos: int) -> float:
        """根据舵机ID和位置计算并应用角度转换规则"""
        angle = (pos - 500) * 180 / 2000  # 基础角度计算
        if servo_id == 2:
            return 180 - angle  # 舵机2: 180度减去基础角度
        elif servo_id == 3:
            return angle - 22.5  # 舵机3: 基础角度减去22.5度
        else:
            return angle  # 其他舵机: 使用基础角度

    def _send_cmd(self, servo_id: int, position: int, time_ms: int):
        """发送指令到舵机（简化日志，不显示中间角度）"""
        cmd = [
            0x55, 0x55,          # 帧头
            8,                   # 数据长度
            0x03,                # 舵机转动指令
            1,                   # 舵机数量
            time_ms & 0xFF,      # 时间低8位
            (time_ms >> 8) & 0xFF,
            servo_id,            # 舵机ID
            position & 0xFF,     # 位置低8位
            (position >> 8) & 0xFF
        ]
        self.ser.write(bytearray(cmd))
        # 简化发送指令的日志，不显示角度（仅在完成后显示最终角度）
        print(f"发送指令: 舵机 {servo_id}，位置 {position}，时间 {time_ms}ms")

    def _angle_to_pos(self, angle: float) -> int:
        """将角度(0-180度)转换为位置值(500-2500)"""
        pos = int(500 + angle * 2000 / 180)
        return max(500, min(2500, pos))

    def control_servo(self, servo_id: int, position: int, total_time_ms: int = 2000):
        """控制单个舵机（平滑分步控制，仅显示最终角度反馈）"""
        current_pos = self.current_pos.get(servo_id, position)
        
        print(f"\n===== 舵机 {servo_id} 控制开始 =====")
        print(f"初始位置: {current_pos}")
        print(f"目标位置: {position}")
        
        # 自动分配步数
        if abs(position - current_pos) < 100:
            steps = 5
        elif abs(position - current_pos) < 500:
            steps = 8
        else:
            steps = 10
            
        step_time_ms = total_time_ms // steps
        step_pos = (position - current_pos) / steps
        
        # 执行平滑控制（不显示中间角度）
        for i in range(1, steps + 1):
            target_pos = int(current_pos + step_pos * i)
            self._send_cmd(servo_id, target_pos, step_time_ms)
            self.current_pos[servo_id] = target_pos
            time.sleep(step_time_ms / 1000)
        
        # 完成后计算并显示最终角度（应用转换规则）
        final_angle = self._get_display_angle(servo_id, position)
        print(f"===== 舵机 {servo_id} 控制完成 =====")
        print(f"最终位置: {position}")
        print(f"最终角度: {final_angle:.1f}°\n")

    def control_servo_by_angle(self, servo_id: int, angle: float, total_time_ms: int = 2000):
        """控制单个舵机（角度值，平滑控制）"""
        position = self._angle_to_pos(angle)
        print(f"舵机 {servo_id} 目标角度: {angle}° → 对应位置: {position}")
        self.control_servo(servo_id, position, total_time_ms)

    def control_multiple_servos(self, actions: List[Tuple[int, int]], total_time_ms: int = 2000):
        """顺序控制多个舵机（仅显示最终角度反馈）"""
        print(f"\n===== 开始顺序控制多个舵机（共 {len(actions)} 个） =====")
        for i, (servo_id, pos) in enumerate(actions, 1):
            print(f"\n----- 第 {i}/{len(actions)} 个舵机 -----")
            self.control_servo(servo_id, pos, total_time_ms)

    def control_multiple_servos_by_angle(self, actions: List[Tuple[int, float]], total_time_ms: int = 2000):
        """顺序控制多个舵机（角度值，仅显示最终角度反馈）"""
        print(f"\n===== 开始顺序控制多个舵机（共 {len(actions)} 个） =====")
        for i, (servo_id, angle) in enumerate(actions, 1):
            print(f"\n----- 第 {i}/{len(actions)} 个舵机 -----")
            print(f"目标角度: {angle}°")
            self.control_servo_by_angle(servo_id, angle, total_time_ms)


# main函数保持不变（参数解析部分）
def main():
    parser = argparse.ArgumentParser(description='舵机控制器（仅显示最终角度反馈，0°=500，180°=2500）')
    parser.add_argument('--port', default='/dev/ttyUSB1', help='串口路径')
    parser.add_argument('--baudrate', type=int, default=9600, help='波特率')

    subparsers = parser.add_subparsers(dest='command', required=True)

    # 1. 控制单个舵机（位置值）
    parser_single = subparsers.add_parser('single', help='控制单个舵机（位置值，范围500-2500）')
    parser_single.add_argument('id', type=int, help='舵机ID（1-255）')
    parser_single.add_argument('position', type=int, help='目标位置（500=0°，2500=180°）')
    parser_single.add_argument('--time', type=int, default=500, help='动作总时间（ms）')

    # 2. 控制单个舵机（角度值）
    parser_single_angle = subparsers.add_parser('single_angle', help='控制单个舵机（角度值，范围0-180°）')
    parser_single_angle.add_argument('id', type=int, help='舵机ID（1-255）')
    parser_single_angle.add_argument('angle', type=float, help='目标角度（0-180°，对应位置500-2500）')
    parser_single_angle.add_argument('--time', type=int, default=500, help='动作总时间（ms）')

    # 3. 顺序控制多个舵机（位置值）
    parser_multi = subparsers.add_parser('multi', help='顺序控制多个舵机（位置值）')
    parser_multi.add_argument('actions', nargs='+', help='舵机动作（格式：ID1:位置1 ID2:位置2 ...）')
    parser_multi.add_argument('--time', type=int, default=500, help='统一动作总时间（ms）')

    # 4. 顺序控制多个舵机（角度值）
    parser_multi_angle = subparsers.add_parser('multi_angle', help='顺序控制多个舵机（角度值）')
    parser_multi_angle.add_argument('actions', nargs='+', help='舵机动作（格式：ID1:角度1 ID2:角度2 ...）')
    parser_multi_angle.add_argument('--time', type=int, default=500, help='统一动作总时间（ms）')

    args = parser.parse_args()

    try:
        controller = ServoController(port=args.port, baudrate=args.baudrate)

        if args.command == 'single':
            controller.control_servo(args.id, args.position, args.time)
        
        elif args.command == 'single_angle':
            controller.control_servo_by_angle(args.id, args.angle, args.time)
        
        elif args.command == 'multi':
            actions = []
            for arg in args.actions:
                try:
                    sid, pos = arg.split(':')
                    actions.append((int(sid), int(pos)))
                except ValueError:
                    print(f"参数错误：{arg}，正确格式应为 ID:位置")
                    return
            controller.control_multiple_servos(actions, args.time)
        
        elif args.command == 'multi_angle':
            actions = []
            for arg in args.actions:
                try:
                    sid, angle = arg.split(':')
                    actions.append((int(sid), float(angle)))
                except ValueError:
                    print(f"参数错误：{arg}，正确格式应为 ID:角度")
                    return
            controller.control_multiple_servos_by_angle(actions, args.time)

    except Exception as e:
        print(f"错误：{str(e)}")
    finally:
        if 'controller' in locals():
            controller.close()


if __name__ == "__main__":
    main()