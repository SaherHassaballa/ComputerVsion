import asyncio
import logging
from mavsdk import System

drone = System()

drone.connect(
        system_address="udpin://192.168.43.135:14550"
    )  # await drone.connect(system_address="serial://COM8:57600")



