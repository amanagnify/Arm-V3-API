cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(robot_arm_v3)

target_sources(app PRIVATE
    src/main.c
    src/servo.c
    src/servo_control.c
    src/wifi_manager.c
    src/tcp_transport.c
    src/websocket_layer.c
    src/app_callbacks.c
)
