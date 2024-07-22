import pandas as pd
from ydata_profiling import ProfileReport


def main():
    machine_failure = pd.read_csv("data/ai4i2020.csv")
    profile = ProfileReport(machine_failure, title="Machine Failure Profiling Report")
    profile.to_file("machine_failure_profiling_report.html")


if __name__ == "__main__":
    main()
