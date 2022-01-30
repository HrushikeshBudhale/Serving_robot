[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Serving_robot

## Table of Contents
   * [What is this?](#what-is-this)
   * [Requirements](#requirements)
   * [Execution](#execution)


## What is this?
Mobile manipulator project demonstrating inverse kinematics and trajectory following capabilities in the CoppeliaSim.

This project utilizes forward, inverse kinematics for UR5 manipulator and 4 wheel differential drive path following on custom designed chassis to serve cups to the tables in the narrow space of diner environment.

<p align="center">
  <img src="https://github.com/HrushikeshBudhale/Serving_robot/blob/main/docs/serving_robot.gif?raw=true" alt="ACTIVITY DIAGRAM" width="600"/>
</p>

- complete demo [video](https://youtu.be/yH5fl2JwCGo)

## Requirements
- Ubuntu18.04 (or higher)
- CoppeliaSim simulator
- python3.6

## Execution

Clone this repository using 
```
git clone git@github.com:HrushikeshBudhale/Serving_robot.git
```
Start the CoppeliaSim simulator by entering
```
<CoppeliaSim-folder-path>/coppeliaSim.sh <serving-robot-folder-path>/scene/diner.ttt
```
In another terminal enter
```
python3 <serving-robot-folder-path>/src/robot_control.py
```
