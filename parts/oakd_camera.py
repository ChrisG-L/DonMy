import depthai as dai
import cv2
from threading import Thread

class OakDCamera:
    def __init__(self, width=160, height=120, fps=20):
        self.outSz = (width, height)
        self.pipeline = dai.Pipeline()
        cam = self.pipeline.create(dai.node.ColorCamera)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
        cam.setIspScale(1, 3)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam.setFps(fps)

        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("rgb")
        cam.isp.link(xout.input)

        self.device = dai.Device(self.pipeline)
        self.q = self.device.getOutputQueue("rgb", 4, False)

        self.frame = None
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            packet = self.q.tryGet()
            if packet is not None:
                img = packet.getCvFrame()
                self.frame = cv2.resize(img, self.outSz, interpolation=cv2.INTER_AREA)

    def run_threaded(self):
        return self.frame

    def run(self):
        return self.frame

    def shutdown(self):
        self.running = False
        self.device.close()

