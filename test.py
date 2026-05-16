import os

from click import clear
clear()
os.getcwd()
print(os.listdir(os.getcwd()))
print(list(os.walk(os.getcwd())))