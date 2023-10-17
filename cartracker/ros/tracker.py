from typing import Any, List, Optional, Tuple
from cartracker.ros.simulation import Simulation
import pandas as pd


class Sensor:
    def __init__(self, name: str, lat: float, lon: float, rad: float) -> None:
        self.name = name
        self.latitude = lat
        self.longitude = lon
        self.radius = rad

    def __repr__(self) -> str:
        return f"Sensor({self.name}, [{self.latitude}, {self.longitude}])"

    def check(self, lat: float, lon: float):
        distance = (self.latitude - lat) ** 2 + (self.longitude - lon) ** 2
        return self.radius**2 > distance


class CarTracker:
    def __init__(self, simulation: Simulation, cctv_list_path: str) -> None:
        self.cctv_list = pd.read_excel(cctv_list_path)
        self.sensor_list: List[Sensor] = []
        for row in self.cctv_list.iloc:
            sensor = Sensor(*row, 0.0015)
            self.sensor_list.append(sensor)
        self.simulation = simulation
        self.simulation.add_callback(self.run_step)

        self.tracking_data = {
            key: [] for key in ["Start_Datetime", "End_Datetime", "Sensor"]
        }
        self.prev_result: Optional[Sensor] = None
        self.start_event: Optional[Tuple[pd.Timestamp, Sensor]] = None

    def run_step(
        self, datetime: pd.Timestamp, lat: float, lon: float
    ) -> Tuple[str, bool]:
        result: Optional[Sensor] = None
        for sensor in self.sensor_list:
            if sensor.check(lat, lon):
                result = sensor

        if result != self.prev_result:
            if result is not None:
                self.start_event = (datetime, result)
            else:
                self.tracking_data["Start_Datetime"].append(self.start_event[0])
                self.tracking_data["End_Datetime"].append(datetime)
                self.tracking_data["Sensor"].append(self.start_event[1].name)

        self.prev_result = result

        return ("Is_Sensored", result.name if result is not None else None)

    def run(self):
        result = self.simulation.simulate()
        tracking_list = pd.DataFrame(self.tracking_data)

        temp = tracking_list.sort_values("Start_Datetime", ignore_index=True)
        temp["Date"] = temp["Start_Datetime"].dt.strftime("%Y-%m-%d")
        temp["Time"] = temp["Start_Datetime"].dt.strftime("%H:%M:%S")
        temp["Start_Datetime"] = temp["Start_Datetime"].astype(str)
        temp["End_Datetime"] = temp["End_Datetime"].astype(str)
        temp[["Start_Datetime", "End_Datetime", "Date", "Time", "Sensor"]].to_excel(
            "./output/result.xlsx"
        )

        return result, tracking_list
