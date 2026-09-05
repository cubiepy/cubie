"""Compile a host-only metadata DLL with explicit installed providers."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).parent
CUDA = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
VC = Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.50.35717")
KIT = Path("C:/Program Files (x86)/Windows Kits/10")
TOOLS = VC / "bin/Hostx64/x64"
INCLUDES = [VC / "include", KIT / "Include/10.0.26100.0/ucrt",
            KIT / "Include/10.0.26100.0/um", KIT / "Include/10.0.26100.0/shared",
            CUDA / "include", CUDA / "extras/CUPTI/include"]
LIBS = [VC / "lib/x64", KIT / "Lib/10.0.26100.0/ucrt/x64",
        KIT / "Lib/10.0.26100.0/um/x64", CUDA / "extras/CUPTI/lib64"]


def asset(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


commands = [
    [str(TOOLS / "cl.exe"), "/nologo", "/c", "/std:c++17", "/EHsc", "/MD", "/O2", "/W4",
     *["/I" + str(path) for path in INCLUDES],
     "/sourceDependencies", str(ROOT / "dependencies.json"),
     "/Fo" + str(ROOT / "collector.obj"), str(ROOT / "collector.cpp")],
    [str(TOOLS / "link.exe"), "/nologo", "/DLL", "/MACHINE:X64",
     "/OUT:" + str(ROOT / "collector.dll"),
     *["/LIBPATH:" + str(path) for path in LIBS],
     str(ROOT / "collector.obj"), str(CUDA / "extras/CUPTI/lib64/cupti.lib")],
    [str(TOOLS / "dumpbin.exe"), "/DEPENDENTS", "/EXPORTS", str(ROOT / "collector.dll")],
]
receipt = dict(status="HOST_BUILD_STARTED", gpu_compilations=0, gpu_launches=0,
               source=asset(ROOT / "collector.cpp"), commands=[])
try:
    for arguments in commands:
        result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True)
        receipt["commands"].append(dict(arguments=arguments, exit=result.returncode,
                                        stdout=result.stdout, stderr=result.stderr))
        result.check_returncode()
    dependency = json.loads((ROOT / "dependencies.json").read_text(encoding="utf-8-sig"))
    headers = dependency["Data"]["Includes"]
    receipt["headers"] = [asset(path) for path in headers]
    receipt["providers"] = [asset(path) for path in (
        TOOLS / "cl.exe", TOOLS / "link.exe", TOOLS / "dumpbin.exe",
        CUDA / "extras/CUPTI/lib64/cupti.lib",
        CUDA / "extras/CUPTI/lib64/cupti64_2026.2.1.dll")]
    receipt["artifacts"] = [asset(ROOT / name) for name in
                            ("collector.obj", "collector.dll", "dependencies.json")]
    receipt["status"] = "HOST_ONLY_DLL_BUILD_PASS_PENDING_INDEPENDENT_REVIEW"
except Exception:
    receipt["status"] = "HOST_BUILD_FAILURE_RETAINED"
    raise
finally:
    (ROOT / "build_receipt.json").write_text(json.dumps(receipt, indent=2))
print(receipt["status"], asset(ROOT / "collector.dll")["sha256"])
