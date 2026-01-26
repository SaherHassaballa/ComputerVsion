import asyncio
from mavsdk import System
import serial.tools.list_ports
import sys

def list_ports():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
    else:
        print("Available serial ports:")
        for p in ports:
            print(f"  {p.device} - {p.description}")

async def try_connect(port, baud, timeout=10):
    addr = f"serial:///{port}:{baud}"
    print(f"\nTrying {addr} (timeout {timeout}s)...")
    drone = System()
    try:
        await drone.connect(system_address=addr)
    except Exception as ex:
        print("connect() raised:", ex)
        return False

    # wait for connection with timeout
    connected = False
    try:
        end = asyncio.get_event_loop().time() + timeout
        async for state in drone.core.connection_state():
            if state.is_connected:
                print(f"Connected to system (sysid {state.uuid}) via {addr}")
                connected = True
                break
            if asyncio.get_event_loop().time() > end:
                print("Timed out waiting for connection_state.")
                break
            await asyncio.sleep(0.1)
    except Exception as ex:
        print("Error while waiting connection_state:", ex)
        return False

    return connected, drone if connected else (False, None)

async def main():
    list_ports()

    # change to the port you see in Device Manager
    port = "COM8"
    # try several baud rates
    bauds = [115200, 57600, 921600]
    for b in bauds:
        ok, drone = await try_connect(port, b, timeout=8)
        if ok:
            # got connection — show telemetry example, then exit
            print("Waiting for global position (GPS) health...")
            async for health in drone.telemetry.health():
                print("Health:", health)
                if health.is_global_position_ok:
                    print("GPS lock acquired!")
                    break
            print("Reading one position sample...")
            async for pos in drone.telemetry.position():
                print(f"lat {pos.latitude_deg:.7f}, lon {pos.longitude_deg:.7f}, alt {pos.absolute_altitude_m:.2f}")
                break
            return
    print("\nAll baud attempts failed. Check device, cable, drivers, and QGroundControl ability to connect.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
