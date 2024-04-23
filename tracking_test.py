from cartracker_v2.tracker.simple_tracker import Tracker, recalc_time

# tracker = Tracker()
# tracker.run()
# tracker.close()

import pandas as pd
data = pd.read_excel("./outputs/result_data.xlsx")
data = recalc_time(data)
data.to_excel("./outputs/result_data.xlsx", index=False)