import threading
import os

threatshold_imu_lean = 50
threatshold_imu_shake = 90
imu_data_init = None
shelf_lean = False
shelf_shake = False

temperature = None
humidity = None
light = None

# Thread lock for loadcell data access
imu_data_init_lock = threading.Lock()
threatshold_imu_lean_lock = threading.Lock()
threatshold_imu_shake_lock = threading.Lock()
temperature_lock = threading.Lock()
humidity_lock = threading.Lock()
light_lock = threading.Lock()
shelf_lean_lock = threading.Lock()
shelf_shake_lock = threading.Lock()


# Last data reception timestamp for connection tracking
last_data_reception_time = 0

# Thread-safe access to global variables
def get_imu_data_init():
    """Get a thread-safe snapshot of imu data"""
    with imu_data_init_lock:
        return imu_data_init

def set_imu_data_init(new_data):
    """Set imu data in a thread-safe way"""
    with imu_data_init_lock:
        global imu_data_init
        imu_data_init = new_data

def get_threatshold_imu_lean():
    """Get a thread-safe snapshot of threatshold_imu_lean"""
    with threatshold_imu_lean_lock:
        return threatshold_imu_lean
    
def set_threatshold_imu_lean(new_threatshold):
    """Set threatshold_imu_lean in a thread-safe way"""
    with threatshold_imu_lean_lock:
        global threatshold_imu_lean
        threatshold_imu_lean = new_threatshold

def get_threatshold_imu_shake():
    """Get a thread-safe snapshot of threatshold_imu_shake"""
    with threatshold_imu_shake_lock:
        return threatshold_imu_shake

def set_threatshold_imu_shake(new_threatshold):
    """Set threatshold_imu_shake in a thread-safe way"""
    with threatshold_imu_shake_lock:
        global threatshold_imu_shake
        threatshold_imu_shake = new_threatshold

def get_temperature():
    """Get a thread-safe snapshot of temperature"""
    with temperature_lock:
        return temperature

def set_temperature(new_temperature):
    """Set temperature in a thread-safe way"""
    with temperature_lock:
        global temperature
        temperature = new_temperature

def get_humidity():
    """Get a thread-safe snapshot of humidity"""
    with humidity_lock:
        return humidity

def set_humidity(new_humidity):
    """Set humidity in a thread-safe way"""
    with humidity_lock:
        global humidity
        humidity = new_humidity

def get_light():
    """Get a thread-safe snapshot of light"""
    with light_lock:
        return light

def set_light(new_light):
    """Set light in a thread-safe way"""
    with light_lock:
        global light
        light = new_light

def get_shelf_lean():
    """Get a thread-safe snapshot of shelf_lean"""
    with shelf_lean_lock:
        return shelf_lean

def set_shelf_lean(new_shelf_lean):
    """Set shelf_lean in a thread-safe way"""
    with shelf_lean_lock:
        global shelf_lean
        shelf_lean = new_shelf_lean
    
def get_shelf_shake():
    """Get a thread-safe snapshot of shelf_shake"""
    with shelf_shake_lock:
        return shelf_shake  
    
def set_shelf_shake(new_shelf_shake):
    """Set shelf_shake in a thread-safe way"""
    with shelf_shake_lock:
        global shelf_shake
        shelf_shake = new_shelf_shake