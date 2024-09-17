import subprocess
import os
from pathlib import Path
from balsam.config import ClientSettings, InvalidSettings, Settings, SiteConfig, site_builder
from balsam.cmdline.site import start, stop
from balsam.api import ApplicationDefinition


def create_site(name, site_path, selected_site):
    cmd = f"balsam site init -n {name} {site_path}"
    ret = subprocess.run(cmd, shell=True, check=True)
    print(ret)

def start_site(abs_site_path):
    cwd = os.getcwd()
    os.chdir(abs_site_path)
    try:
        cmd = f"balsam site start"
        ret = subprocess.run(cmd, shell=True, check=True)
        print(ret)
    except:
        cmd = f"balsam site sync"
        ret = subprocess.run(cmd, shell=True, check=True)
        print(ret)
    os.chdir(cwd)



def register_app():
    print("registering app now")
    class BenchmarkApp(ApplicationDefinition):
        site = "benchmark"
        
        def run(self, app_path, args):
            import sys
            sys.path.append(app_path)
            from apps import sleeper
            return sleeper(**args)
    BenchmarkApp.sync()

def balsam_initialize(site_path=Path("benchmark"), name="benchmark"):

    #make_site_cmd = "balsam site init -n benchmark benchmark"
    #ret = subprocess.run(make_site_cmd)
    selected_site = "alcf_polaris"
    abs_site_path = site_path.absolute()
    print(f"{site_path=}")
    if not Path(os.path.join(abs_site_path, "settings.yml")).exists():
        create_site(name, site_path, selected_site)
    start_site(abs_site_path)
    register_app()
    return

if __name__ == '__main__':
    
    balsam_initialize()
    
