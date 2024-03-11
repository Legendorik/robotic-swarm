import subprocess
import threading
import psutil
from time import sleep

argos_path = 'argos3'
argos_experiment_path = './argos/footbot-ai.argos'


class Argos:
    def __init__(self):
        self.argos_process = None
        self.setup()

    def setup(self):
        self.argos_process = subprocess.Popen([argos_path, '-c', argos_experiment_path], stdout=subprocess.PIPE)
        # self.argos_process = subprocess.Popen([argos_path, '-z', '-c', argos_experiment_path], stdout=subprocess.PIPE)
        sleep(3) # wait for argos to start
        threading.Thread(target=self.start, daemon=True).start()

    def start(self):
        while not self.argos_process.poll():
            #print("Running...")
            sleep(.1)

    def kill(self):
        # a = self.argos_process.stdout.readlines()
        # print(a)
        self.argos_process.kill()
        

    def memory_usage(self, pid):
        proc = psutil.Process(pid)
        mem = proc.memory_info().rss  # resident memory
        for child in proc.children(recursive=True):
            try:
                mem += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return mem
