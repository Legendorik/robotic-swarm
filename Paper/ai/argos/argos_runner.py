import mmap
import os
import subprocess
import threading
import psutil
from time import sleep
from typing import List

argos_path = 'argos3'
argos_experiment_path = './argos/footbot-ai.argos'
argos_foraging_path = './argos/foraging.argos'
data_fname = '_data_robot_'
actions_fname = '_actions_robot_'
file_size = 128


class Argos:
    created_instances = 0
    def __init__(self, num_robots=1, render_mode='human', verbose=False):
        self.argos_process = None
        self.num_robots = num_robots
        self.data_mmaps: List[mmap.mmap] = []
        self.actions_mmaps : List[mmap.mmap] = []
        self.verbose = verbose
        self.render_mode = render_mode
        self.files_id = Argos.created_instances
        Argos.created_instances = Argos.created_instances + 1
        self.setup()

    def setup(self):
        self.create_files()
        self.create_mappings()
        
        env = os.environ.copy()
        env['FILES_ID'] = str(self.files_id)

        if (self.render_mode == 'human'):
            self.argos_process = subprocess.Popen([argos_path, '-l', f'./log_{self.files_id}.txt', '-e', f'./logerr_{self.files_id}.txt', '-c', argos_foraging_path], stdout=subprocess.PIPE, env=env)
            sleep(3) # wait for argos to start
            
        else:
            # self.argos_process = subprocess.Popen([argos_path, '-z', '-c', argos_experiment_path], stdout=subprocess.PIPE) # При запуске из питона аргос постоянно отправляет одни и те же наблюдения. При запуске аргоса отдельно все работает
            self.argos_process = subprocess.Popen([argos_path, '-z', '-l', f'./log_{self.files_id}.txt', '-e', f'./logerr_{self.files_id}.txt', '-c', argos_foraging_path], stdout=subprocess.PIPE,env=env)
            sleep(.25) # wait for argos to start
            # print("PID"self.argos_process.pid)
            
        
        
        # threading.Thread(target=self.start, daemon=True).start()

    def create_files(self):
        files_id = str(self.files_id)
        for i in range(self.num_robots):
            data_file = files_id + data_fname + str(i)
            actions_file = files_id + actions_fname + str(i)
            if not os.path.isfile(data_file):
                with open(data_file, "w+b") as fd:
                    fd.write(b'\0' * file_size)
            if not os.path.isfile(actions_file):
                with open(actions_file, "w+b") as fd:
                    fd.write(b'\0' * file_size)
    
    def create_mappings(self):
        files_id = str(self.files_id)
        for i in range(self.num_robots):
            data_file = files_id + data_fname + str(i)
            actions_file = files_id + actions_fname + str(i)
            with open(data_file, "r+b") as fd:
                fd.write(b'\0' * file_size)
                self.data_mmaps.append(mmap.mmap(fd.fileno(), file_size, access=mmap.ACCESS_READ, offset=0))
            with open(actions_file, "r+b") as fd:
                fd.write(b'\0' * file_size)
                self.actions_mmaps.append(mmap.mmap(fd.fileno(), file_size, access=mmap.ACCESS_WRITE, offset=0))

    def send_to(self, message, robot_id=0):
        extra_space = file_size - len(message)
        modified_message = message + '\0' * extra_space
        bytes_message = bytes(modified_message, 'ASCII')
        self.actions_mmaps[robot_id].seek(0)
        self.actions_mmaps[robot_id].write(bytes_message)
        if self.verbose:
            print("MSG out P --> A[", robot_id, "]: ", message)

    def receive_from(self, robot_id=0):
        self.data_mmaps[robot_id].seek(0)
        read_result = self.data_mmaps[robot_id].readline()
        message = read_result.decode('ASCII')
        if self.verbose:
            print("MSG in P <-- A[", robot_id, "]: ", message)
        return message

    def start(self):
        while not self.argos_process.poll():
            #print("Running...")
            sleep(.1)

    def kill(self):
        # a = self.argos_process.stdout.readlines()
        # print(a)
        files_id = str(self.files_id)
        self.argos_process.kill()
        for i in range(self.num_robots):
            data_file = files_id + data_fname + str(i)
            actions_file = files_id + actions_fname + str(i)
            self.data_mmaps[i].close()
            self.actions_mmaps[i].close()
            os.remove(data_file)
            os.remove(actions_file)
        

    def memory_usage(self, pid):
        proc = psutil.Process(pid)
        mem = proc.memory_info().rss  # resident memory
        for child in proc.children(recursive=True):
            try:
                mem += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return mem
