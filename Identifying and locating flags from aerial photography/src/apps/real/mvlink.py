#!/usr/bin/env python3

import asyncio
import logging
from mavsdk import System

# Enable INFO level logging by default so that INFO messages are shown
logging.basicConfig(level=logging.INFO)

async def run():
    # Init the drone
    drone = System()
    await drone.connect(
        system_address="udpin://192.168.43.135:14550"
    )  # await drone.connect(system_address="serial://COM8:57600")

    # Event to control exit
    exit_event = asyncio.Event()

    # Start telemetry tasks
    _tasks = [
        asyncio.create_task(print_battery(drone)),
        asyncio.create_task(print_gps_info(drone)),
        asyncio.create_task(print_in_air(drone)),
        asyncio.create_task(print_position(drone)),
        asyncio.create_task(stop_on_input(exit_event)),  # 👈 New stop task
    ]

    # Keep the program running until event is set
    await exit_event.wait()

    print("Stopping tasks...")
    for t in _tasks:
        t.cancel()
    await asyncio.gather(*_tasks, return_exceptions=True)
    print("Program stopped cleanly ✅")

async def stop_on_input(event):
    """Wait for user to press Enter, then stop program"""
    await asyncio.to_thread(input, "\nPress Enter to stop...\n")
    event.set()

async def print_battery(drone):
    async for battery in drone.telemetry.battery():
        print(f"Battery: {battery.remaining_percent:.2f}")

async def print_gps_info(drone):
    async for gps_info in drone.telemetry.gps_info():
        print(f"GPS info: {gps_info}")

async def print_in_air(drone):
    async for in_air in drone.telemetry.in_air():
        print(f"In air: {in_air}")

async def print_position(drone):
    async for position in drone.telemetry.position():
        print(position)

if __name__ == "__main__":
    asyncio.run(run())
