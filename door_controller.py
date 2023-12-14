import time
import requests


class DoorController():
    def __init__(self, url, wait_time=5) -> None:
        self.checker = ["", "", "", "", ""]
        self.url = url
        self.wait_time = wait_time
        self.open_timer = 0
    
    def open(self, name):
        if (time.time() - self.open_timer) < self.wait_time:
            return
        print(f"Open door for {name}")
        response = requests.get(self.url)
        self.open_timer = time.time()
        
    def visit(self, text):
        name = "No Person" if text == None else text
        self.checker.append(name)
        self.checker.pop(0)
        if name != "Guest" and name != "No Person":
            if self.checker.count(name) > 3:
                self.open(name)
                return True, name
        return False, name