from djitellopy import Tello
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TelloController:
    def __init__(self):
        self.tello = Tello()
        self.connected = False
        self.logger = logging.getLogger(__name__)

    def connect(self):
        try:
            self.logger.info("Connecting to Tello")
            self.tello.connect()
            self.connected = True
            self.logger.info(f"Tello Battery: {self.tello.get_battery()}%")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to Tello: {e}")
            return False

    def takeoff(self):
        self.logger.info("Taking off")
        self.tello.takeoff()

    def land(self):
        self.logger.info("Landing")
        self.tello.land()

    def send_rc_control(self, left_right, forward_backward, up_down, yaw):
        self.logger.info(f"Maneuvering: LR={left_right}, FB={forward_backward}, UD={up_down}, Yaw={yaw}")
        self.tello.send_rc_control(left_right, forward_backward, up_down, yaw)
