"""Administrator helper: grant one user access to currently connected ZED devices."""

import argparse
import os
import pwd
import shlex
import stat
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--user", default="sangjunpark")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
uid = pwd.getpwnam(args.user).pw_uid
devices = set()
for folder in ("/sys/class/video4linux", "/sys/class/hidraw", "/sys/bus/usb/devices"):
    for entry in Path(folder).glob("*"):
        resolved = entry.resolve()
        if not any(
            (parent / "idVendor").is_file() and (parent / "idVendor").read_text().strip() == "2b03"
            for parent in (resolved, *resolved.parents)
        ):
            continue
        for line in (entry / "uevent").read_text().splitlines():
            if line.startswith("DEVNAME="):
                device = Path("/dev") / line.split("=", 1)[1]
                if device.exists() and stat.S_ISCHR(device.stat().st_mode):
                    devices.add(str(device))
if not devices:
    raise SystemExit("No connected Stereolabs device nodes found")
command = ["setfacl", "-m", f"u:{uid}:rw", *sorted(devices)]
print(shlex.join(command), flush=True)
if not args.dry_run:
    if os.geteuid() != 0:
        raise SystemExit("An administrator must run this helper with sudo; --dry-run changes nothing")
    subprocess.run(command, check=True)
    print(
        f"Granted {args.user} access to {len(devices)} ZED device nodes. Retry capture in the existing session."
    )
