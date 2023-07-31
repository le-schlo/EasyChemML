import copy
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from EasyChemML.JobSystem.CheckpointSystem.CheckpointSystem import CheckpointSystem


class Environment():
    TMP_path: str
    WORKING_path: str
    CHECKPOINT_path: str

    CheckpointSystem: CheckpointSystem

    def __init__(self, TMP_path_addRelativ: str = None, WORKING_path_addRelativ: str = None, CHECKPOINT_path:str = None,
                 TMP_path: str = None, WORKING_path: str = None):
        if TMP_path is None:
            self.TMP_path = self._generate_TMP_path()
        else:
            self.TMP_path = TMP_path

        if WORKING_path is None:
            self.WORKING_path = self._generate_WORKING_path()
        else:
            self.WORKING_path = WORKING_path

        if CHECKPOINT_path is None:
            self.CHECKPOINT_path = self._generate_CHECKPOINT_path()
        else:
            self.CHECKPOINT_path = CHECKPOINT_path

        if TMP_path_addRelativ is not None:
            self.TMP_path = os.path.join(self.TMP_path, TMP_path_addRelativ)

        if WORKING_path_addRelativ is not None:
            self.WORKING_path = os.path.join(self.WORKING_path, WORKING_path_addRelativ)

        if not os.path.exists(self.TMP_path):
            os.mkdir(self.TMP_path)

        if not os.path.exists(self.WORKING_path):
            os.mkdir(self.WORKING_path)

        if not os.path.exists(self.CHECKPOINT_path):
            os.mkdir(self.CHECKPOINT_path)

        self._createLogging()
        self._changeCPUAffinity()
        self.CheckpointSystem = CheckpointSystem(self)

    def _changeCPUAffinity(self):
        if not os.name == 'nt':  # not possible for windows
            affinity_mask = range(os.cpu_count())
            os.sched_setaffinity(os.getpid(), affinity_mask)

    def clean(self):
        time.sleep(1)
        try:
            shutil.rmtree(self.TMP_path)
        except:
            print('Cleanup failed')

    def _createLogging(self):
        from importlib import reload
        reload(logging)

        logging.basicConfig(filename=os.path.join(self.WORKING_path, 'MainNode.log'), level=logging.INFO)
        logging.info(' ###################################################################################')
        logging.info('    ______           _______     _______ _    _ ______ __  __        __  __ _       ')
        logging.info('   |  ____|   /\    / ____\ \   / / ____| |  | |  ____|  \/  |      |  \/  | |      ')
        logging.info('   | |__     /  \  | (___  \ \_/ / |    | |__| | |__  | \  / |______| \  / | |      ')
        logging.info('   |  __|   / /\ \  \___ \  \   /| |    |  __  |  __| | |\/| |______| |\/| | |      ')
        logging.info('   | |____ / ____ \ ____) |  | | | |____| |  | | |____| |  | |      | |  | | |____  ')
        logging.info('   |______/_/    \_\_____/   |_|  \_____|_|  |_|______|_|  |_|      |_|  |_|______| ')
        logging.info('####################################################################################')

        print(' ###################################################################################')
        print('    ______           _______     _______ _    _ ______ __  __        __  __ _       ')
        print('   |  ____|   /\    / ____\ \   / / ____| |  | |  ____|  \/  |      |  \/  | |      ')
        print('   | |__     /  \  | (___  \ \_/ / |    | |__| | |__  | \  / |______| \  / | |      ')
        print('   |  __|   / /\ \  \___ \  \   /| |    |  __  |  __| | |\/| |______| |\/| | |      ')
        print('   | |____ / ____ \ ____) |  | | | |____| |  | | |____| |  | |      | |  | | |____  ')
        print('   |______/_/    \_\_____/   |_|  \_____|_|  |_|______|_|  |_|      |_|  |_|______| ')
        print('####################################################################################')

    def _generate_TMP_path(self):
        programm_path, file = os.path.split(sys.argv[0])
        tmp_path = os.path.join(programm_path, 'TMP')

        if os.path.exists(tmp_path):
            print('remove TMP')
            try:
                shutil.rmtree(tmp_path)
            except:
                print('delete tmp folder failed')
            time.sleep(0.1)

        try:
            os.mkdir(tmp_path)
        except:
            print('tmp folder already exists')
        return tmp_path

    def _generate_WORKING_path(self):
        programm_path, file = os.path.split(sys.argv[0])
        return os.path.join(programm_path)

    def _generate_CHECKPOINT_path(self):
        return os.path.join(self.WORKING_path, 'Checkpoints')
