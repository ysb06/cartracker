from typing import Dict


def run_simulation(config: Dict):
    from .simulation import Simulation
    from .tracker import CarTracker

    simulation = Simulation(config["data_path"]["bags"])
    tracker = CarTracker(simulation, config["data_path"]["cctv"])

    raws, _ = tracker.run()
    for idx, (key, val) in enumerate(raws.items()):
        val["Datetime"] = val["Datetime"].astype(str)
        val.to_excel(f"./output/raws/{idx}.xlsx")
