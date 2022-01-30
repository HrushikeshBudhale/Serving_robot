import sim
import time
import sys
import numpy as np
import sympy as sp

'''
#################################### Setup #####################################
'''
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
if clientID!=-1:
    print ('Connected to remote API server')
else:
	print('connection not successful')
	sys.exit("could not connect")

emptyBuff = bytearray()  # empty buffer



# Defining Symbols
th1, th2, th3, th4, th5, th6 = sp.symbols('\\theta_1^*, \\theta_2^*, \\theta_3^*, \\theta_4^*, \\theta_5^*, \\theta_6^*')

# Defining Constants
d1 = 0.089159
d4 = 0.10915
d5 = 0.09465
d6 = 0.0823

a2 = -0.425
a3 = -0.39225


def getJacobianUR5():
    J=sp.Matrix([[-a2*sp.sin(th1)*sp.sin(th2) - d6*sp.sin(th1)*sp.sin(th5)*sp.cos(th2 + th3 + th4) - a3*sp.sin(th1)*sp.sin(th2 + th3) + d5*sp.sin(th1)*sp.sin(th2 + th3 + th4) + d6*sp.cos(th1)*sp.cos(th5) + d4*sp.cos(th1), 
                -(d6*sp.sin(th5)*sp.sin(th2 + th3 + th4) - a2*sp.cos(th2) - a3*sp.cos(th2 + th3) + d5*sp.cos(th2 + th3 + th4))*sp.cos(th1), 
                -(d6*sp.sin(th5)*sp.sin(th2 + th3 + th4) - a3*sp.cos(th2 + th3) + d5*sp.cos(th2 + th3 + th4))*sp.cos(th1), 
                -(d6*sp.sin(th5)*sp.sin(th2 + th3 + th4) + d5*sp.cos(th2 + th3 + th4))*sp.cos(th1), 
                -d6*sp.sin(th1)*sp.sin(th5) + d6*sp.cos(th1)*sp.cos(th5)*sp.cos(th2 + th3 + th4), 
                0],
                [d6*sp.sin(th1)*sp.cos(th5) + d4*sp.sin(th1) + a2*sp.sin(th2)*sp.cos(th1) + d6*sp.sin(th5)*sp.cos(th1)*sp.cos(th2 + th3 + th4) + a3*sp.sin(th2 + th3)*sp.cos(th1) - d5*sp.sin(th2 + th3 + th4)*sp.cos(th1), 
                -(d6*sp.sin(th5)*sp.sin(th2 + th3 + th4) - a2*sp.cos(th2) - a3*sp.cos(th2 + th3) + d5*sp.cos(th2 + th3 + th4))*sp.sin(th1), 
                -(d6*sp.sin(th5)*sp.sin(th2 + th3 + th4) - a3*sp.cos(th2 + th3) + d5*sp.cos(th2 + th3 + th4))*sp.sin(th1), 
                -(d6*sp.sin(th5)*sp.sin(th2 + th3 + th4) + d5*sp.cos(th2 + th3 + th4))*sp.sin(th1), 
                d6*sp.sin(th1)*sp.cos(th5)*sp.cos(th2 + th3 + th4) + d6*sp.sin(th5)*sp.cos(th1), 
                0], 
                [0, 
                a2*sp.sin(th2) + d6*sp.sin(th5)*sp.cos(th2 + th3 + th4) + a3*sp.sin(th2 + th3) - d5*sp.sin(th2 + th3 + th4), 
                d6*sp.sin(th5)*sp.cos(th2 + th3 + th4) + a3*sp.sin(th2 + th3) - d5*sp.sin(th2 + th3 + th4), 
                d6*sp.sin(th5)*sp.cos(th2 + th3 + th4) - d5*sp.sin(th2 + th3 + th4), 
                d6*sp.sin(th2 + th3 + th4)*sp.cos(th5), 
                0], 
                [0, sp.sin(th1), sp.sin(th1), sp.sin(th1), -sp.sin(th2 + th3 + th4)*sp.cos(th1), sp.sin(th1)*sp.cos(th5) + sp.sin(th5)*sp.cos(th1)*sp.cos(th2 + th3 + th4)], 
                [0, -sp.cos(th1), -sp.cos(th1), -sp.cos(th1), -sp.sin(th1)*sp.sin(th2 + th3 + th4), sp.sin(th1)*sp.sin(th5)*sp.cos(th2 + th3 + th4) - sp.cos(th1)*sp.cos(th5)], 
                [1, 0, 0, 0, sp.cos(th2 + th3 + th4), sp.sin(th5)*sp.sin(th2 + th3 + th4)]])
    return J

'''
############################## Class Definitions ###############################
'''

class Cart:
    def __init__(self):
        self.wheelRadius = 0.1
        self.wheelSeperation = 0.6
        self.wMax = 4.5
        self.vicinityThresh = 0.0
        self.pathFollowingStatus = "pause"

        retval, self.frontLeftJointHandle = sim.simxGetObjectHandle(clientID, "leftFrontJoint", sim.simx_opmode_oneshot_wait)
        retval, self.frontRightJointHandle = sim.simxGetObjectHandle(clientID, "rightFrontJoint", sim.simx_opmode_oneshot_wait)
        retval, self.rearRightJointHandle = sim.simxGetObjectHandle(clientID, "rightRearJoint", sim.simx_opmode_oneshot_wait)
        retval, self.rearLeftJointHandle = sim.simxGetObjectHandle(clientID, "leftRearJoint", sim.simx_opmode_oneshot_wait)
        retval, self.cartHandle = sim.simxGetObjectHandle(clientID, "cart", sim.simx_opmode_oneshot_wait)
        retval, self.cartBaseHandle = sim.simxGetObjectHandle(clientID, "cartBase", sim.simx_opmode_oneshot_wait)

    def get_position(self, objectHandle=-1, withRefTo=-1):
        if objectHandle == -1:
            retval, pos = sim.simxGetObjectPosition(clientID, self.cartHandle, withRefTo, sim.simx_opmode_oneshot_wait)
        else:
            retval, pos = sim.simxGetObjectPosition(clientID, objectHandle, self.cartHandle, sim.simx_opmode_oneshot_wait)
            return np.array([round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)])

    def set_wheel_speed(self, velLeft=0, velRight=0):
        velLeft  = np.clip(velLeft,  -self.wMax, self.wMax)
        velRight = np.clip(velRight, -self.wMax, self.wMax)
        sim.simxSetJointTargetVelocity(clientID, self.frontLeftJointHandle, -velRight, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(clientID, self.rearLeftJointHandle, velRight, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(clientID, self.frontRightJointHandle, -velLeft, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(clientID, self.rearRightJointHandle, velLeft, sim.simx_opmode_oneshot)

    def set_target_object(self, objectName):
        retval, self.targetHandle = sim.simxGetObjectHandle(clientID, objectName, sim.simx_opmode_oneshot_wait)

    def setPathFollowingStatus(self, status="pause"):
        # status: "pause" or "resume"
        self.pathFollowingStatus = status
        retval = sim.simxCallScriptFunction(clientID, "PathFollowerDummy", sim.sim_scripttype_childscript, 
                                                        "setPathFollowingStatus", 
                                                        [], 
                                                        [], 
                                                        [status], 
                                                        emptyBuff, 
                                                        sim.simx_opmode_oneshot_wait)
        return retval[0]

    def goToTable(self,tableNumber):
        print(f"Going to table{tableNumber}")
        retval, tableHandle = sim.simxGetObjectHandle(clientID, f"Path__ctrlPt{tableNumber+1}", sim.simx_opmode_oneshot_wait)
        targetPose = self.get_position(self.targetHandle)
        self.setPathFollowingStatus("resume")
        self.vicinityThresh = 0.0
        while True:
            if self.pathFollowingStatus == "resume":
                targetPose = self.get_position(self.targetHandle)
                if self._isNear(self.targetHandle, withRefTo=tableHandle, thresh=0.05):
                    self.vicinityThresh = 0.3
                    self.setPathFollowingStatus("pause")
            
            w_l, w_r = self._compute_reqd_velocity(targetPose)
            self.set_wheel_speed(w_l, w_r)
            if self._isNear(self.cartBaseHandle, withRefTo=self.targetHandle, thresh=self.vicinityThresh):
                break
        self.set_wheel_speed(0.0, 0.0)
        print(f"Reached at table{tableNumber}")
        return

    def _isNear(self, objectHandle, withRefTo=-1, thresh=0.2):
        retval, position = sim.simxGetObjectPosition(clientID, objectHandle, withRefTo, sim.simx_opmode_oneshot_wait)
        # print("position: ", position)
        # print("error: ", np.linalg.norm(np.array(position)))
        if np.linalg.norm(np.array(position)) < thresh:
            return True
        return False

    def _compute_reqd_velocity(self, targetPose):
        distance = np.linalg.norm(targetPose)
        phi = np.arctan2(targetPose[1], targetPose[0])
        # Proportional control
        v_des = 1 * np.clip(distance, -0.5, 0.5)
        w_des = 3 * phi
        v_r = v_des - (self.wheelSeperation/2)*w_des
        v_l = v_des + (self.wheelSeperation/2)*w_des
        # calc wheel speed
        w_r = v_r/self.wheelRadius
        w_l = v_l/self.wheelRadius
        return w_r, w_l


class Manipulator:
    def __init__(self, goToInitPos=True):
        self.gripper = Gripper()
        self.targetPose = np.array([0.0]*6)

        # Joint angles in radian
        self.actualJointAngles = np.array([0.0]*6)
        self.targetJointAngles = np.array([0.0, 0.0, np.pi/2, 0.0, -np.pi/2, 0.0])
        self.initialConfig = np.array([0.0, 0.0, np.pi/2, 0.0, -np.pi/2, 0.0])
        self.intermediateConfig = np.array([0.0, 0.0, np.pi/2, 0.0, -np.pi/2, np.pi/2])

        retval, self.endEffectorHandle = sim.simxGetObjectHandle(clientID, "UR5_tip", sim.simx_opmode_oneshot_wait)
        retval, self.armBaseHandle = sim.simxGetObjectHandle(clientID, "UR5", sim.simx_opmode_oneshot_wait)

        if goToInitPos:
            # Set initial configuration
            self.moveToConfig(config=self.initialConfig, kinematics="FK")

        self.jointHandles = [-1]*6
        for i in range(6):
            retval, self.jointHandles[i] = sim.simxGetObjectHandle(clientID, f"UR5_joint{i+1}", sim.simx_opmode_oneshot_wait)
        
    def getPose(self, ObjectName="", withRefTo=-1):
        if ObjectName == "":
            # return endEffector pose wrt base
            retval, position = sim.simxGetObjectPosition(clientID, self.endEffectorHandle, withRefTo, sim.simx_opmode_oneshot_wait)
            retval, orientation = sim.simxGetObjectQuaternion(clientID, self.endEffectorHandle, withRefTo, sim.simx_opmode_oneshot_wait)
        else:
            retval, goalHandle = sim.simxGetObjectHandle(clientID, ObjectName, sim.simx_opmode_oneshot_wait)
            retval, position = sim.simxGetObjectPosition(clientID, goalHandle, withRefTo, sim.simx_opmode_oneshot_wait)
            retval, orientation = sim.simxGetObjectQuaternion(clientID, goalHandle, withRefTo, sim.simx_opmode_oneshot_wait)
        goalPose = [position[0], position[1], position[2], orientation[0], orientation[1], orientation[2], orientation[3]]
        return goalPose

    def placeObject(self, objectName, toLocationName):
        goalPose = self.getPose(objectName)
        self.moveToConfig(config=goalPose, kinematics="IK")
        self.gripper.pick(objectName)
        goalPose = self.getPose(toLocationName)
        self.moveToConfig(config=self.intermediateConfig, kinematics="FK")
        self.moveToConfig(config=goalPose, kinematics="IK")
        self.gripper.placeOn(toLocationName)
        self.moveToConfig(config=self.initialConfig, kinematics="FK")
        return

    def pickObject(self, objectName, placeOnCart):
        goalPose = self.getPose(objectName)
        self.moveToConfig(config=goalPose, kinematics="IK")
        self.gripper.pick(objectName)
        goalPose = self.getPose(placeOnCart)
        self.moveToConfig(config=self.intermediateConfig, kinematics="FK")
        self.moveToConfig(config=goalPose, kinematics="IK")
        self.gripper.placeOn("cart")
        self.moveToConfig(config=self.initialConfig, kinematics="FK")
        return

    def moveToPose(self, targetPose):
        self.targetPose = targetPose
        dP = sp.Matrix([[0.], [0.], [0.], [0.], [0.], [0.]])
        J = getJacobianUR5()
        for i in range(6):
            retval, self.actualJointAngles[i] = sim.simxGetJointPosition(clientID, self.jointHandles[i], sim.simx_opmode_oneshot_wait)
        
        while True:
            if self._isTargetPositionReached(self.armBaseHandle):
                break
            P_tip = self.getPose("", withRefTo=self.armBaseHandle)  # pose of end effector wrt base
            
            dP[0] = np.clip(targetPose[0] - P_tip[0], -0.1, 0.1)
            dP[1] = np.clip(targetPose[1] - P_tip[1], -0.1, 0.1)
            dP[2] = np.clip(targetPose[2] - P_tip[2], -0.1, 0.1)
            dP[3] = 0.0
            dP[4] = 0.0
            dP[5] = 0.0
            
            Js = J.subs([(th1, self.actualJointAngles[0]), (th2, self.actualJointAngles[1]), (th3, self.actualJointAngles[2]), 
                         (th4, self.actualJointAngles[3]), (th5, self.actualJointAngles[4]), (th6, self.actualJointAngles[5])])
            J_inv = Js.inv('LU')
            dTh = np.array(J_inv @ dP).reshape(1,6)
            dTh = np.clip(dTh, -0.05, 0.05)
            jointConfig = self.actualJointAngles + dTh
            self.moveToJointConfig(jointConfig[0])
        return
       
    def _isTargetPositionReached(self, withRefTo=-1):
        retval, endEffectorPosition = sim.simxGetObjectPosition(clientID, self.endEffectorHandle, withRefTo, sim.simx_opmode_oneshot_wait)
        endEffectorPosition = np.array(endEffectorPosition)
        # print("===========================")
        # print("targetPose: ", self.targetPose[:3])
        # print("P_tip:      ", endEffectorPosition[:3])
        # print("error: ",np.linalg.norm(self.targetPose[:3] - endEffectorPosition))
        # print("===========================")
        if np.linalg.norm(self.targetPose[:3] - endEffectorPosition) < 0.02:
            return True
        return False
        
    def moveToJointConfig(self, jointConfig):
        for i in range(6):
            sim.simxSetJointTargetPosition(clientID, self.jointHandles[i], jointConfig[i], sim.simx_opmode_oneshot)
        while not self._isJointTargetPositionReached(jointConfig):
            pass
        return

    def _isJointTargetPositionReached(self, jointConfig):
        for i in range(6):
            retval, self.actualJointAngles[i] = sim.simxGetJointPosition(clientID, self.jointHandles[i], sim.simx_opmode_oneshot_wait)
        # print("===========================")
        # print("actualAngles: ", self.actualJointAngles)
        # print("JointConfig:      ", jointConfig)
        # print("error: ", jointConfig - self.actualJointAngles)
        # print("===========================")
        if max(jointConfig - self.actualJointAngles) < 0.01:
            return True
        return False

    def moveToConfig(self, config, kinematics="IK"):
        if kinematics not in ["IK", "FK"]:
            print("Wrong kinematic type given")
            return False
        elif kinematics == "IK":
            config[2] += 0.15  # gripper should be 15 cm above the object
            config[3] = 180 # gripper should be upside down
            config[4] = 180
        retval = sim.simxCallScriptFunction(clientID, "UR5", sim.sim_scripttype_childscript, 
                                                        "startMovingArm", 
                                                        [], 
                                                        config, 
                                                        [kinematics], 
                                                        emptyBuff, 
                                                        sim.simx_opmode_oneshot_wait)
        print("Going to Checkpoint")
        time.sleep(2)
        while not self._isArmReached():
            time.sleep(1)
        print(f"Reached.")
        return

    def _isArmReached(self):
        retval = sim.simxCallScriptFunction(clientID, "UR5", sim.sim_scripttype_childscript, 
                                                        "isArmReached", 
                                                        [], 
                                                        [], 
                                                        "", 
                                                        emptyBuff, 
                                                        sim.simx_opmode_oneshot_wait)
        return retval[1][0] == 1


class Gripper:
    def __init__(self):
        retval, self.grip_top = sim.simxGetObjectHandle(clientID, 'ROBOTIQ_85_active1', sim.simx_opmode_blocking)
        retval, self.grip_bottom = sim.simxGetObjectHandle(clientID, 'ROBOTIQ_85_active2', sim.simx_opmode_blocking)
        retval, self.gripT_pos = sim.simxGetJointPosition(clientID, self.grip_top, sim.simx_opmode_streaming)
        retval, self.gripB_pos = sim.simxGetJointPosition(clientID, self.grip_bottom, sim.simx_opmode_streaming)
        retval, self.connector=sim.simxGetObjectHandle(clientID,'ROBOTIQ_85', sim.simx_opmode_blocking)

        self.grippedObject = None
        
    def openCloseGripper(self, activity="close"):
        if activity == "close":
            activity = "closeClicked"
            print("Gripper closed")
        else:
            activity = "openClicked"
            print("Gripper opened")
        retval = sim.simxCallScriptFunction(clientID, "ROBOTIQ_85", sim.sim_scripttype_childscript, 
                                                    activity,'','', 
                                                    "", 
                                                    emptyBuff, 
                                                    sim.simx_opmode_oneshot_wait)
        return

    def gripObject(self):
        sim.simxSetJointTargetVelocity(clientID, self.grip_top, 0, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(clientID, self.grip_bottom, 0, sim.simx_opmode_oneshot)

    def pick(self, objectName):
        #  self.openCloseGripper("close")
        time.sleep(2)
        self.gripObject()
        retval, objectHandle = sim.simxGetObjectHandle(clientID, objectName, sim.simx_opmode_blocking)
        sim.simxSetObjectParent(clientID, objectHandle, self.connector, True, sim.simx_opmode_blocking)
        self.grippedObject = objectHandle
        return

    def placeOn(self, parentName=""):
        self.openCloseGripper("open")
        if self.grippedObject == None:
            print("No object in gripper")
        retval, parentHandle = sim.simxGetObjectHandle(clientID, parentName, sim.simx_opmode_blocking)
        sim.simxSetObjectParent(clientID, self.grippedObject, parentHandle, True, sim.simx_opmode_blocking)
        return
        
'''
################################## Main logic ##################################
'''
    
def main():
    try:
        retval = sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot_wait)
        time.sleep(1) # necessary for simulation to start properly

        # # Manipulator demo
        # # Using custom functions for solving IK
        # arm = Manipulator(goToInitPos=False)
        # arm.moveToJointConfig(arm.targetJointAngles)
        # for waypoint in [1, 2, 3, 4]:
        #     goalPose = arm.getPose(f"waypoint{waypoint}", withRefTo=arm.armBaseHandle)
        #     arm.moveToPose(targetPose=goalPose)
        #     print(f"waypoint{waypoint} reached")
        # arm.moveToJointConfig(arm.targetJointAngles)
        
        # Complete demo
        cart = Cart()
        arm = Manipulator()

        cart.set_target_object("PathFollowerDummy")

        # Order 1 Placing
        cart.goToTable(0)
        arm.placeObject("Cup0", "Table1PlaceLocation")

        # Order 2 Picking
        cart.goToTable(1)
        arm.pickObject("Cup4", "CartPlaceLocation1")

        # Order 3 Placing
        cart.goToTable(2)
        arm.placeObject("Cup1", "Table3PlaceLocation")
        
        # Order 4 Placing
        cart.goToTable(4)
        arm.placeObject("Cup2", "Table5PlaceLocation")
        
        # Order 5 Picking
        cart.goToTable(5)
        arm.pickObject("Cup3", "CartPlaceLocation2")

        print("---- All orders fulfilled ----")
        
    except Exception as e:
        print(e)
    finally:
        retval = sim.simxPauseSimulation(clientID,sim.simx_opmode_oneshot_wait)


if __name__ == "__main__":
    main()